"""
Tests for `deep_gemm.build_gather_layout_for_rank_overlap`.

The generator builds (gather_index, tile_rank, grouped_layout, m_logical)
from a global MoE routing_topk in **layout A** (expert-major outer +
ring-order rank-minor inner). See
`docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md` (§11) for the spec.

We compare the GPU output against a pure-Python reference implementation
of the same layout. The two should agree on:

  - `m_logical` (scalar, exact)
  - `tile_rank[: num_m_tiles]` (exact)
  - `grouped_layout[: m_logical]` (exact)
  - `gather_index[: m_logical]` (multiset per (expert, ring-step) chunk;
    pad rows have value `-1`; the order of real-row tokens within a chunk
    may differ between GPU and Python because Phase 3 uses `atomicAdd`).

cd /home/guanrui03/cursor/moe/MoE_DeepGEMM_Combine/DeepGEMM
PYTHONPATH=$PWD python tests/test_gather_layout_generator.py            # 跑全部 (unit + e2e)
PYTHONPATH=$PWD python tests/test_gather_layout_generator.py -t unit    # 只跑 unit (layout 逐元素对齐参考)
PYTHONPATH=$PWD python tests/test_gather_layout_generator.py -t e2e     # 只跑 e2e (generator → grouped GEMM)
"""

import argparse
import os
import shutil
import sys

cache_dir = os.path.expanduser("~/.deep_gemm/cache")
if os.path.exists(cache_dir):
    shutil.rmtree(cache_dir)
    print("Clearing DeepGEMM JIT cache...")

import torch

import deep_gemm
from deep_gemm.testing import get_arch_major


# ---------------------------------------------------------------------------
# Pure-Python reference implementation (layout A)
# ---------------------------------------------------------------------------
def reference_build_gather_layout(routing_topk: torch.Tensor,
                                  local_rank: int,
                                  num_ranks: int,
                                  tokens_per_rank: int,
                                  num_experts: int,
                                  block_m: int):
    """Mirror of `deep_gemm.build_gather_layout_for_rank_overlap`.

    Returns: (gather_index, tile_rank, grouped_layout, m_logical) where
    `gather_index` and `grouped_layout` have length `m_logical` (i.e.
    truncated, no trailing slack).
    """
    assert routing_topk.dim() == 2
    T, K = routing_topk.shape
    assert T == num_ranks * tokens_per_rank
    rt_cpu = routing_topk.detach().cpu().numpy()

    # routing_topk holds GLOBAL expert ids; this rank owns the contiguous block
    # [expert_base, expert_base + num_experts). Mirror the kernel's global->local
    # map + drop-others filter exactly (Phase 1 / Phase 3 stay in lockstep).
    expert_base = local_rank * num_experts

    counts = [[0] * num_ranks for _ in range(num_experts)]
    for t in range(T):
        r = t // tokens_per_rank
        for j in range(K):
            e = int(rt_cpu[t, j]) - expert_base
            if 0 <= e < num_experts:
                counts[e][r] += 1

    starts = [[0] * num_ranks for _ in range(num_experts)]
    tile_rank_list: list[int] = []
    cum = 0
    for e in range(num_experts):
        for s in range(num_ranks):
            r = (local_rank + s) % num_ranks
            starts[e][s] = cum
            n_real = counts[e][r]
            n_slot = ((n_real + block_m - 1) // block_m) * block_m
            tile_rank_list.extend([r] * (n_slot // block_m))
            cum += n_slot
    m_logical = cum

    gather = [-1] * m_logical
    glayout = [0] * m_logical
    cursor = [[0] * num_ranks for _ in range(num_experts)]
    for t in range(T):
        r = t // tokens_per_rank
        s = (r - local_rank + num_ranks) % num_ranks
        for j in range(K):
            e = int(rt_cpu[t, j]) - expert_base
            if not (0 <= e < num_experts):
                continue
            off = cursor[e][s]
            cursor[e][s] += 1
            pos = starts[e][s] + off
            gather[pos] = t
            glayout[pos] = e
    # Pad-row grouped_layout: chunk's expert id (matches GPU Phase 2 fill).
    for e in range(num_experts):
        for s in range(num_ranks):
            r = (local_rank + s) % num_ranks
            n_real = counts[e][r]
            n_slot = ((n_real + block_m - 1) // block_m) * block_m
            base = starts[e][s]
            for i in range(n_real, n_slot):
                glayout[base + i] = e

    device = routing_topk.device
    return (
        torch.tensor(gather, dtype=torch.int32, device=device),
        torch.tensor(tile_rank_list, dtype=torch.int32, device=device),
        torch.tensor(glayout, dtype=torch.int32, device=device),
        m_logical,
    )


# ---------------------------------------------------------------------------
# Per-chunk multiset comparison for gather_index
# ---------------------------------------------------------------------------
def _chunk_multiset_equal(gpu_gather: torch.Tensor,
                          ref_gather: torch.Tensor,
                          ref_grouped_layout: torch.Tensor,
                          ref_tile_rank: torch.Tensor,
                          counts_cpu,
                          local_rank: int,
                          num_ranks: int,
                          num_experts: int,
                          block_m: int):
    """Compare gather_index as a multiset per (expert, ring-step) chunk.

    Within one chunk the GPU's atomicAdd order is non-deterministic, so we
    sort each chunk's real-row slice on both sides and compare. Pad-row
    positions (gather == -1) must match exactly position-for-position.
    """
    cum = 0
    for e in range(num_experts):
        for s in range(num_ranks):
            r = (local_rank + s) % num_ranks
            n_real = counts_cpu[e][r]
            n_slot = ((n_real + block_m - 1) // block_m) * block_m
            real_slice = slice(cum, cum + n_real)
            pad_slice = slice(cum + n_real, cum + n_slot)

            gpu_real = gpu_gather[real_slice].sort().values
            ref_real = ref_gather[real_slice].sort().values
            if not torch.equal(gpu_real, ref_real):
                return (False,
                        f"real-row multiset mismatch at e={e}, s={s}, r={r}: "
                        f"gpu={gpu_real.tolist()[:8]}..., ref={ref_real.tolist()[:8]}...")

            gpu_pad = gpu_gather[pad_slice]
            if (gpu_pad != -1).any():
                bad = (gpu_pad != -1).nonzero().flatten().tolist()
                return (False,
                        f"pad-row not -1 at e={e}, s={s}: bad offsets {bad[:8]}...")

            cum += n_slot
    return True, ""


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
def _generate_routing_topk(num_total_tokens: int, top_k: int,
                           num_experts: int, *, seed: int = 0):
    """Random top-k expert choices with no per-token duplicates."""
    g = torch.Generator(device='cuda').manual_seed(seed)
    # `torch.randperm` per row is the cleanest way to get distinct top-k.
    out = torch.empty((num_total_tokens, top_k), dtype=torch.int32, device='cuda')
    for t in range(num_total_tokens):
        out[t] = torch.randperm(num_experts, generator=g, device='cuda')[:top_k].to(torch.int32)
    return out


def _format_table(title: str, headers, rows):
    if not rows:
        return
    cell_rows = [[str(c) for c in r] for r in rows]
    widths = [max(len(h), *(len(r[i]) for r in cell_rows)) for i, h in enumerate(headers)]
    sep = ' | '
    header_line = sep.join(f'{h:>{w}}' for h, w in zip(headers, widths))
    bar = '=' * len(header_line)
    print(); print(bar); print(f'  {title}'); print(bar)
    print(header_line); print('-' * len(header_line))
    for r in cell_rows:
        print(sep.join(f'{c:>{w}}' for c, w in zip(r, widths)))
    print(bar)


def test_gather_layout_generator():
    print('Testing build_gather_layout_for_rank_overlap (Phase 1+2+3 generator):')
    print('  Compares GPU output vs pure-Python reference (layout A).')
    print()

    cases = [
        # (num_ranks, tokens_per_rank, num_experts, top_k, block_m)
        ( 1,   64,   8, 1, 64),    # rank=1 → ring is trivial; small
        ( 1,  256,  16, 2, 64),    # rank=1 with multi-expert
        ( 2,  128,   8, 2, 64),    # smallest multi-rank
        ( 4,  256,  16, 2, 128),   # mid-sized
        ( 4,  512,  32, 4, 128),   # 4-rank with bigger top_k
        ( 8,  256,  64, 4, 128),   # NVL8, 64 experts
        ( 8,  500,  64, 4, 128),   # tokens_per_rank not multiple of block_m
        ( 8, 1024,  64, 4, 128),   # large
        ( 8,  256,  64, 6, 128),   # high top_k
    ]

    summary = []
    for num_ranks, tpr, num_experts, top_k, block_m in cases:
        T = num_ranks * tpr
        if num_experts < top_k:
            continue

        for local_rank in range(num_ranks) if num_ranks <= 4 else [0, num_ranks - 1]:
            torch.manual_seed(0xc0ffee ^ local_rank)
            # routing_topk now holds GLOBAL expert ids drawn from the full pool
            # of `num_experts * num_ranks` experts (num_experts = per-rank count).
            routing_topk = _generate_routing_topk(T, top_k, num_experts * num_ranks,
                                                  seed=0xfade ^ local_rank)

            # GPU
            gpu_gather, gpu_tile, gpu_glayout, gpu_m_t, gpu_psum, gpu_row_topk = \
                deep_gemm.build_gather_layout_for_rank_overlap(
                    routing_topk, local_rank, num_ranks, tpr, num_experts, block_m)
            torch.cuda.synchronize()
            gpu_m = int(gpu_m_t.item())

            # Reference
            ref_gather, ref_tile, ref_glayout, ref_m = reference_build_gather_layout(
                routing_topk, local_rank, num_ranks, tpr, num_experts, block_m)

            # ---- m_logical ----
            assert gpu_m == ref_m, \
                (f'm_logical mismatch: gpu={gpu_m}, ref={ref_m} '
                 f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                 f'top_k={top_k}, block_m={block_m}, local_rank={local_rank})')

            # ---- tile_rank ----
            num_m_tiles = gpu_m // block_m
            gpu_tile_used = gpu_tile[:num_m_tiles]
            assert torch.equal(gpu_tile_used, ref_tile), \
                (f'tile_rank mismatch (first 16): gpu={gpu_tile_used[:16].tolist()}, '
                 f'ref={ref_tile[:16].tolist()} '
                 f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                 f'local_rank={local_rank})')

            # ---- grouped_layout ----
            gpu_glayout_used = gpu_glayout[:gpu_m]
            assert torch.equal(gpu_glayout_used, ref_glayout), \
                (f'grouped_layout mismatch '
                 f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                 f'local_rank={local_rank})')

            # ---- gather_index (per-chunk multiset) ----
            counts_cpu = [[0] * num_ranks for _ in range(num_experts)]
            rt_cpu = routing_topk.cpu().numpy()
            expert_base = local_rank * num_experts
            for t in range(T):
                r = t // tpr
                for j in range(top_k):
                    e = int(rt_cpu[t, j]) - expert_base
                    if 0 <= e < num_experts:
                        counts_cpu[e][r] += 1
            ok, msg = _chunk_multiset_equal(
                gpu_gather[:gpu_m], ref_gather, ref_glayout, ref_tile,
                counts_cpu, local_rank, num_ranks, num_experts, block_m)
            assert ok, (f'gather_index mismatch: {msg} '
                        f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                        f'local_rank={local_rank})')
            gpu_row_topk_used = gpu_row_topk[:gpu_m]
            assert torch.equal(gpu_row_topk_used[gpu_gather[:gpu_m] < 0],
                               torch.full_like(gpu_row_topk_used[gpu_gather[:gpu_m] < 0], -1))
            real_topk = gpu_row_topk_used[gpu_gather[:gpu_m] >= 0]
            assert ((real_topk >= 0) & (real_topk < top_k)).all(), \
                (f'row_to_topk out of range '
                 f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                 f'local_rank={local_rank})')

            n_pad = (gpu_gather[:gpu_m] == -1).sum().item()
            print(f'  > nr={num_ranks}, tpr={tpr:5}, ne={num_experts:3}, k={top_k}, '
                  f'block_m={block_m:3}, lr={local_rank} | m={gpu_m:6}, n_pad={n_pad:5}  OK')
            summary.append((num_ranks, tpr, num_experts, top_k, block_m,
                            local_rank, gpu_m, n_pad))

    _format_table(
        'Summary: gather-layout generator vs pure-Python reference',
        ['ranks', 'tpr', 'experts', 'top_k', 'block_m', 'local', 'm_logical', 'n_pad'],
        summary)


def test_gather_layout_with_gemm():
    """End-to-end: generator → m-grouped GEMM (rank_flags pre-set to 1).

    This is purely an accuracy/wiring smoke check on a single GPU.
    """
    print()
    print('End-to-end: generator → m-grouped GEMM (single-GPU, flags pre-set to 1):')

    sys.path.insert(0, os.path.dirname(__file__))
    from generators import KernelType, MajorTypeAB, QuantConfig, cast_fp8_fp4_with_major, grouped_cast_fp8_fp4_with_major

    quant_config = QuantConfig()
    use_ue8m0 = False
    disable_ue8m0_cast = not use_ue8m0
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    cases = [
        # (num_ranks, tpr, n_experts, top_k, n, k, block_m)
        (4, 256, 8,  2, 4096, 7168, 128),
        (8, 256, 16, 4, 4096, 7168, 128),
    ]

    for num_ranks, tpr, num_experts, top_k, n, k, block_m in cases:
        T = num_ranks * tpr
        for local_rank in [0, num_ranks - 1]:
            torch.manual_seed(0xfeed ^ local_rank)
            # GLOBAL expert ids over the full pool; the layout maps them to this
            # rank's local experts internally.
            routing_topk = _generate_routing_topk(T, top_k, num_experts * num_ranks,
                                                  seed=0xbeef ^ local_rank)

            # Build the layout
            gather_index, tile_rank, grouped_layout, m_logical_t, psum_layout, row_to_topk = \
                deep_gemm.build_gather_layout_for_rank_overlap(
                    routing_topk, local_rank, num_ranks, tpr, num_experts, block_m)
            m_logical = int(m_logical_t.item())

            # Construct A_pool of shape (T, k) and weights B (G, n, k)
            a_pool_bf16 = torch.randn((T, k), device='cuda', dtype=torch.bfloat16)
            b_bf16 = torch.randn((num_experts, n, k), device='cuda', dtype=torch.bfloat16)

            a_pool_fp8 = cast_fp8_fp4_with_major(a_pool_bf16, MajorTypeAB.KMajor,
                                                 quant_config.gran_k_a,
                                                 quant_config.is_fp4_a, use_ue8m0)
            b_fp8 = grouped_cast_fp8_fp4_with_major(b_bf16, MajorTypeAB.KMajor,
                                                    quant_config.gran_k_b,
                                                    quant_config.is_fp4_b, use_ue8m0,
                                                    use_block_cast_for_fp8=True)

            d = torch.empty((m_logical, n), device='cuda', dtype=torch.bfloat16)
            rank_flags = torch.ones(num_ranks, dtype=torch.int64, device='cuda')

            # Reference D[i] = (gather_index[i] >= 0 ? A_pool[gather_index[i]] : 0) @ B[grouped_layout[i]].T
            gi_long = gather_index[:m_logical].long()
            gl_long = grouped_layout[:m_logical].long()
            is_pad = gi_long < 0
            safe_gi = torch.where(is_pad, torch.zeros_like(gi_long), gi_long)
            a_logical = a_pool_bf16[safe_gi].clone()
            a_logical[is_pad] = 0
            ref_d = torch.empty_like(d, dtype=torch.float32)
            for e in range(num_experts):
                mask = (gl_long == e)
                if mask.any():
                    ref_d[mask] = a_logical[mask].float() @ b_bf16[e].float().t()
            ref_d = ref_d.to(torch.bfloat16)

            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                a_pool_fp8, b_fp8, d, grouped_layout[:m_logical],
                disable_ue8m0_cast=disable_ue8m0_cast,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                gather_index=gather_index[:m_logical],
                rank_flags=rank_flags,
                tile_rank=tile_rank[: m_logical // block_m],
                num_ranks=num_ranks,
                rank_flag_epoch=1,
            )
            torch.cuda.synchronize()

            # Pad rows must come out exactly 0.
            if is_pad.any():
                d_pad = d[is_pad]
                assert torch.all(d_pad == 0), \
                    f'pad rows non-zero: ranks={num_ranks}, tpr={tpr}, ne={num_experts}, lr={local_rank}'

            from deep_gemm.testing import calc_diff
            diff = calc_diff(d.to(torch.bfloat16), ref_d)
            assert diff < quant_config.max_diff(), \
                f'e2e diff fails: ranks={num_ranks}, tpr={tpr}, ne={num_experts}, lr={local_rank}, diff={diff:.5f}'

            n_pad = int(is_pad.sum().item())
            print(f'  > nr={num_ranks}, tpr={tpr}, ne={num_experts:3}, k={top_k}, lr={local_rank} '
                  f'| m={m_logical:5}, n_pad={n_pad:5}, diff={diff:.0e}  OK')


def main(argv=None):
    if get_arch_major() != 9:
        print('build_gather_layout_for_rank_overlap is only supported on SM90; '
              'this run is not SM90.')
        return

    parser = argparse.ArgumentParser()
    parser.add_argument('--tests', '-t', nargs='+',
                        choices=['unit', 'e2e', 'all'], default=['all'])
    args = parser.parse_args(argv)
    selected = ['unit', 'e2e'] if 'all' in args.tests else args.tests

    print('Library path:')
    print(f' > {deep_gemm.__path__}\n')

    if 'unit' in selected:
        test_gather_layout_generator()
    if 'e2e' in selected:
        test_gather_layout_with_gemm()


if __name__ == '__main__':
    torch.manual_seed(0)
    main()
