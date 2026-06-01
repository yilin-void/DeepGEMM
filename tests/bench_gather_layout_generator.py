"""
Bench `deep_gemm.build_gather_layout_for_rank_overlap` (Phase 1+2+3 generator).

Measures both:
  - per-kernel GPU time via `bench_kineto` (3 separate kernels in the trace)
  - end-to-end wall time via `bench()` (includes alloc + JIT-cache lookup
    + 3 kernel launches; what the caller actually pays per build).

Default config matches the user's request:
    tokens_per_rank = 7351,  num_ranks = 8,  top_k = 16,
    num_experts     = 512,   block_m   = 128.

Run:
    python tests/bench_gather_layout_generator.py

cd /home/guanrui03/cursor/moe/MoE_DeepGEMM_Combine/DeepGEMM
PYTHONPATH=$PWD python tests/bench_gather_layout_generator.py
"""

import argparse

import torch

# We deliberately import `deep_gemm` from `site-packages` (the just-built
# wheel from `bash install.sh`). The in-tree `deep_gemm/` folder may carry a
# stale `_C.so` from a prior build that doesn't match the current
# `csrc/jit_kernels/impls/moe_gather_layout.hpp`; mixing host code (old) with
# device headers (new) would manifest as a JIT NVCC error like
# `identifier "prefix_and_fill_for_gather_layout" is undefined`.
import deep_gemm
from deep_gemm.testing import bench, bench_kineto, get_arch_major
from deep_gemm.testing.bench import suppress_stdout_stderr


# ---------------------------------------------------------------------------
# Routing-topk generator
# ---------------------------------------------------------------------------
def _generate_routing_topk(num_total_tokens: int, top_k: int,
                           num_experts: int, *, seed: int = 0) -> torch.Tensor:
    """Random distinct top-k per row.

    Uses a single ``argsort`` over a random matrix instead of looping
    ``randperm`` per row, so generating ~60k rows × 16 entries doesn't take
    minutes.
    """
    g = torch.Generator(device='cuda').manual_seed(seed)
    scores = torch.rand((num_total_tokens, num_experts),
                        generator=g, device='cuda')
    return scores.argsort(dim=1)[:, :top_k].contiguous().to(torch.int32)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------
def _format_table(title: str, headers, rows):
    if not rows:
        return
    cell_rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cell_rows))
              for i, h in enumerate(headers)]
    sep = ' | '
    header_line = sep.join(f'{h:>{w}}' for h, w in zip(headers, widths))
    bar = '=' * len(header_line)
    print()
    print(bar)
    print(f'  {title}')
    print(bar)
    print(header_line)
    print('-' * len(header_line))
    for r in cell_rows:
        print(sep.join(f'{c:>{w}}' for c, w in zip(r, widths)))
    print(bar)


def _silent_warmup(fn, n: int = 2):
    """Warm up under suppressed stdout to mute JIT compile chatter."""
    with suppress_stdout_stderr():
        for _ in range(n):
            fn()
    torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# Main bench routine
# ---------------------------------------------------------------------------
def bench_one(num_ranks: int, tokens_per_rank: int, top_k: int,
              num_experts: int, block_m: int,
              num_kineto_tests: int = 30,
              local_ranks=None):
    T = num_ranks * tokens_per_rank
    if local_ranks is None:
        local_ranks = [0]

    # `num_experts` here is the GLOBAL expert count; the API takes the per-rank
    # (local) count and maps global ids internally.
    num_local_experts = num_experts // num_ranks
    assert num_local_experts > 0, \
        f'num_experts({num_experts}) must be >= num_ranks({num_ranks})'

    # One routing topk shared across local_ranks (Phase 1/3 only depend on the
    # topk; Phase 2 outputs depend on `local_rank` but the *cost* doesn't).
    # Values are GLOBAL expert ids drawn from the full pool of `num_experts`.
    torch.manual_seed(0xc0ffee)
    routing_topk = _generate_routing_topk(T, top_k, num_experts, seed=0xfade)

    # Tight analytical M_max used by the generator (see
    # `csrc/jit_kernels/impls/moe_gather_layout.hpp` and §11.4.4 in the design
    # doc). `T*K` is the exact upper bound on Σ n_real (sum of real-row chunk
    # sizes); each non-empty chunk contributes ≤ block_m-1 pad rows on top.
    total_pairs = T * top_k
    num_chunks = num_local_experts * num_ranks
    M_max = total_pairs + min(num_chunks, total_pairs) * (block_m - 1)
    M_max_loose = num_local_experts * num_ranks * \
        (((tokens_per_rank + block_m - 1) // block_m) * block_m)

    print(f'  T = {T} ({num_ranks} × {tokens_per_rank}), '
          f'top_k = {top_k}, num_experts(global) = {num_experts}, '
          f'local_experts = {num_local_experts}, block_m = {block_m}')
    print(f'  M_max (tight, allocated) = {M_max:,} '
          f'({M_max * 4 / 1e6:.1f} MB per int32 tensor)')
    print(f'  M_max (loose, OLD bound) = {M_max_loose:,} '
          f'(would be {M_max_loose * 4 / 1e6:.1f} MB) — '
          f'savings = {M_max_loose / M_max:.1f}×')

    rows_kineto = []
    rows_e2e    = []
    for local_rank in local_ranks:
        def fn():
            return deep_gemm.build_gather_layout_for_rank_overlap(
                routing_topk, local_rank, num_ranks,
                tokens_per_rank, num_local_experts, block_m)

        # First build also reports the actual m_logical for context.
        out = fn()
        torch.cuda.synchronize()
        m_logical = int(out[3].item())
        n_pad = int((out[0][:m_logical] == -1).sum().item())

        # Warm up to fill JIT cache and CUDA contexts.
        _silent_warmup(fn, n=2)

        # 1) Per-kernel GPU time via kineto. Phase 2 was split into 2a (serial
        # prefix, single block) and 2b (per-chunk fill, multi-block) so we now
        # report four kernels.
        t_hist, t_prefix, t_fill, t_scat = bench_kineto(
            fn,
            ('histogram_for_gather_layout',
             'prefix_for_gather_layout',
             'fill_layout_tables_for_gather_layout',
             'scatter_for_gather_layout'),
            num_tests=num_kineto_tests,
            suppress_kineto_output=True,
        )
        t_gpu_total = t_hist + t_prefix + t_fill + t_scat

        # 2) End-to-end wall time (includes alloc + JIT cache lookup +
        # 3 launches + GPU work). This is what the caller pays.
        t_wall = bench(fn, num_warmups=5, num_tests=20)

        rows_kineto.append((
            local_rank, m_logical, n_pad,
            f'{t_hist * 1e6:.1f}',
            f'{t_prefix * 1e6:.1f}',
            f'{t_fill * 1e6:.1f}',
            f'{t_scat * 1e6:.1f}',
            f'{t_gpu_total * 1e6:.1f}',
            f'{t_wall * 1e6:.1f}',
            f'{(t_wall - t_gpu_total) * 1e6:.1f}',
        ))
        rows_e2e.append((
            local_rank, m_logical, n_pad,
            f'{t_wall * 1e6:.1f}',
        ))

    _format_table(
        f'Per-kernel GPU time (kineto, avg of {num_kineto_tests}) — units: us',
        ['local_rank', 'm_logical', 'n_pad',
         'Phase1\nhist',
         'Phase2a\nprefix',
         'Phase2b\nfill',
         'Phase3\nscatter',
         'GPU\ntotal',
         'wall\n(bench)',
         'wall − GPU\n(launch+alloc)'],
        rows_kineto,
    )

    return rows_kineto


def main(argv=None):
    if get_arch_major() != 9:
        print('build_gather_layout_for_rank_overlap is SM90-only; '
              'skip on this GPU.')
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--tokens-per-rank', type=int, default=2*3488)
    parser.add_argument('--num-ranks', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=16)
    parser.add_argument('--num-experts', type=int, default=512)
    parser.add_argument('--block-m', type=int, default=128)
    parser.add_argument('--num-kineto-tests', type=int, default=200)
    parser.add_argument('--all-local-ranks', action='store_true',
                        help='Bench every local_rank in [0, num_ranks). '
                             'By default we only bench local_rank=0 since '
                             'the per-rank cost barely changes.')
    args = parser.parse_args(argv)

    print('Library path:')
    print(f' > {deep_gemm.__path__[0]}')
    dev = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f' > GPU: {dev.name}, {dev.total_memory / 1e9:.0f} GB, '
          f'SM{dev.major}{dev.minor}, {dev.multi_processor_count} SMs')
    print()
    print('Benching `build_gather_layout_for_rank_overlap`:')

    local_ranks = list(range(args.num_ranks)) if args.all_local_ranks else [0]
    bench_one(
        num_ranks=args.num_ranks,
        tokens_per_rank=args.tokens_per_rank,
        top_k=args.top_k,
        num_experts=args.num_experts,
        block_m=args.block_m,
        num_kineto_tests=args.num_kineto_tests,
        local_ranks=local_ranks,
    )


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
