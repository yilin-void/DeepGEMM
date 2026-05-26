"""
Tests for `deep_gemm.build_gather_layout_for_rank_overlap`.

The generator builds (gather_index, tile_rank, grouped_layout, m_logical,
psum_layout, row_to_topk)
from a global MoE routing_topk in **layout A** (expert-major outer +
ring-order rank-minor inner). See
`docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md` (§11) for the spec.

We compare the GPU output against a pure-Python reference implementation
of the same layout. The two should agree on:

  - `m_logical` (scalar, exact)
  - `tile_rank[: num_m_tiles]` (exact)
  - `grouped_layout[: m_logical]` (exact)
  - `psum_layout` (exact)
  - `(gather_index, row_to_topk)[: m_logical]` (pair multiset per
    (expert, ring-step) chunk; pad rows have value `-1`; the order of real rows
    within a chunk may differ between GPU and Python because Phase 3 uses
    `atomicAdd`).
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
                                  block_m: int,
                                  topk_slot_offset: int = 0,
                                  expert_srank_padding: bool = True):
    """Mirror of `deep_gemm.build_gather_layout_for_rank_overlap`.

    Returns: (gather_index, tile_rank, grouped_layout, m_logical, psum_layout,
    row_to_topk) where row-wise tensors have length `m_logical` (i.e.
    truncated, no trailing slack).
    """
    assert routing_topk.dim() == 2
    T, K = routing_topk.shape
    assert T == num_ranks * tokens_per_rank
    rt_cpu = routing_topk.detach().cpu().numpy()

    counts = [[0] * num_ranks for _ in range(num_experts)]  # 每个专家分别从各个rank接收token的个数，shape [num_experts, num_ranks]
    for t in range(T):
        r = t // tokens_per_rank
        for j in range(K):
            e = int(rt_cpu[t, j])
            if 0 <= e < num_experts:
                counts[e][r] += 1

    starts = [[0] * num_ranks for _ in range(num_experts)]  # 每个专家分别从各个rank接收token的开始位置，shape [num_experts, num_ranks]
    tile_rank_list: list[int] = []  # len [num_tile]
    cum = 0
    for e in range(num_experts):
        if expert_srank_padding:
            for s in range(num_ranks):
                r = (local_rank + s) % num_ranks
                starts[e][s] = cum
                n_real = counts[e][r]
                n_slot = ((n_real + block_m - 1) // block_m) * block_m
                tile_rank_list.extend([r] * (n_slot // block_m))
                cum += n_slot
        else:
            expert_start = cum
            n_total = 0
            rank_counts = []
            for s in range(num_ranks):
                r = (local_rank + s) % num_ranks
                starts[e][s] = expert_start + n_total
                n_real = counts[e][r]
                rank_counts.append((r, n_real))
                n_total += n_real
            n_slot = ((n_total + block_m - 1) // block_m) * block_m
            tile_rank_list.extend([-1] * (n_slot // block_m))
            cum += n_slot
    m_logical = cum

    gather = [-1] * m_logical
    row_to_topk = [-1] * m_logical
    glayout = [0] * m_logical   # 表示每个token属于哪个专家
    cursor = [[0] * num_ranks for _ in range(num_experts)]  # 每个专家分别从各个rank已经写了多少真实 row，shape [num_experts, num_ranks]
    for t in range(T):
        r = t // tokens_per_rank
        s = (r - local_rank + num_ranks) % num_ranks    # 表示当前rank下ring第s个位置
        for j in range(K):
            e = int(rt_cpu[t, j])
            if not (0 <= e < num_experts):
                continue
            off = cursor[e][s]  # 当前chunk写入了一些token，off记录下一个写入的位置
            cursor[e][s] += 1
            pos = starts[e][s] + off
            gather[pos] = t
            row_to_topk[pos] = topk_slot_offset + j
            glayout[pos] = e
    # Pad-row grouped_layout: chunk/expert's expert id (matches GPU Phase 2 fill).
    for e in range(num_experts):
        if expert_srank_padding:
            for s in range(num_ranks):
                r = (local_rank + s) % num_ranks
                n_real = counts[e][r]
                n_slot = ((n_real + block_m - 1) // block_m) * block_m
                base = starts[e][s]
                for i in range(n_real, n_slot):
                    glayout[base + i] = e
        else:
            base = starts[e][0]
            next_base = starts[e + 1][0] if e + 1 < num_experts else m_logical
            for i in range(next_base - base):
                glayout[base + i] = e

    psum_layout = []    # 表示专家在token维度上的结束边界
    for e in range(num_experts):
        psum_layout.append(starts[e + 1][0] if e + 1 < num_experts else m_logical)

    device = routing_topk.device
    return (
        torch.tensor(gather, dtype=torch.int32, device=device),
        torch.tensor(tile_rank_list, dtype=torch.int32, device=device),
        torch.tensor(glayout, dtype=torch.int32, device=device),
        m_logical,
        torch.tensor(psum_layout, dtype=torch.int32, device=device),
        torch.tensor(row_to_topk, dtype=torch.int32, device=device),
    )


# ---------------------------------------------------------------------------
# Per-chunk multiset comparison for gather_index + row_to_topk
# ---------------------------------------------------------------------------
def _chunk_pair_multiset_equal(gpu_gather: torch.Tensor,
                               gpu_row_topk: torch.Tensor,
                               ref_gather: torch.Tensor,
                               ref_row_topk: torch.Tensor,
                               counts_cpu,
                               local_rank: int,
                               num_ranks: int,
                               num_experts: int,
                               block_m: int,
                               expert_srank_padding: bool):
    """Compare (gather_index, row_to_topk) pairs per chunk.

    Within one chunk the GPU's atomicAdd order is non-deterministic, so we
    sort each chunk's real-row pairs on both sides and compare. Pad-row
    positions must be exactly (-1, -1).
    """
    cum = 0
    for e in range(num_experts):
        if not expert_srank_padding:
            expert_start = cum
            n_total = 0
            for s in range(num_ranks):
                r = (local_rank + s) % num_ranks
                n_real = counts_cpu[e][r]
                real_slice = slice(expert_start + n_total,
                                   expert_start + n_total + n_real)
                gpu_real = sorted(zip(gpu_gather[real_slice].cpu().tolist(),
                                      gpu_row_topk[real_slice].cpu().tolist()))
                ref_real = sorted(zip(ref_gather[real_slice].cpu().tolist(),
                                      ref_row_topk[real_slice].cpu().tolist()))
                if gpu_real != ref_real:
                    return (False,
                            f"real-row pair multiset mismatch at e={e}, s={s}, r={r}: "
                            f"gpu={gpu_real[:8]}..., ref={ref_real[:8]}...")
                n_total += n_real

            n_slot = ((n_total + block_m - 1) // block_m) * block_m
            pad_slice = slice(expert_start + n_total, expert_start + n_slot)
            gpu_pad = gpu_gather[pad_slice]
            gpu_pad_topk = gpu_row_topk[pad_slice]
            if ((gpu_pad != -1) | (gpu_pad_topk != -1)).any():
                bad = ((gpu_pad != -1) | (gpu_pad_topk != -1)).nonzero().flatten().tolist()
                return (False,
                        f"expert pad-row not (-1, -1) at e={e}: bad offsets {bad[:8]}...")
            cum += n_slot
            continue

        for s in range(num_ranks):
            r = (local_rank + s) % num_ranks
            n_real = counts_cpu[e][r]
            n_slot = ((n_real + block_m - 1) // block_m) * block_m
            real_slice = slice(cum, cum + n_real)
            pad_slice = slice(cum + n_real, cum + n_slot)

            gpu_real = sorted(zip(gpu_gather[real_slice].cpu().tolist(),
                                  gpu_row_topk[real_slice].cpu().tolist()))
            ref_real = sorted(zip(ref_gather[real_slice].cpu().tolist(),
                                  ref_row_topk[real_slice].cpu().tolist()))
            if gpu_real != ref_real:
                return (False,
                        f"real-row pair multiset mismatch at e={e}, s={s}, r={r}: "
                        f"gpu={gpu_real[:8]}..., ref={ref_real[:8]}...")

            gpu_pad = gpu_gather[pad_slice]
            gpu_pad_topk = gpu_row_topk[pad_slice]
            if ((gpu_pad != -1) | (gpu_pad_topk != -1)).any():
                bad = ((gpu_pad != -1) | (gpu_pad_topk != -1)).nonzero().flatten().tolist()
                return (False,
                        f"pad-row not (-1, -1) at e={e}, s={s}: bad offsets {bad[:8]}...")

            cum += n_slot
    return True, ""


# ---------------------------------------------------------------------------
# Test driver
# ---------------------------------------------------------------------------
def _generate_routing_topk(num_total_tokens: int, top_k: int,
                           num_experts: int, *, seed: int = 0):
    """Random top-k expert choices with no per-token duplicates."""
    if top_k > num_experts:
        raise ValueError(f'top_k ({top_k}) must be <= num_experts ({num_experts})')
    g = torch.Generator(device='cuda').manual_seed(seed)
    # Vectorized top-k keeps the large overlap-benchmark shape fast enough for
    # unit testing while preserving distinct expert choices per token.
    scores = torch.rand((num_total_tokens, num_experts),
                        generator=g, device='cuda')
    return torch.topk(scores, top_k, dim=1).indices.contiguous().to(torch.int32)


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
        ( 8, 6976,  64, 2, 128),   # overlap e2e shape: global_experts=512, local_top_k=2
    ]

    summary = []
    for num_ranks, tpr, num_experts, top_k, block_m in cases:
        T = num_ranks * tpr
        if num_experts < top_k:
            continue

        for expert_srank_padding in (True, False):
            for local_rank in range(num_ranks) if num_ranks <= 4 else [0, num_ranks - 1]:
                torch.manual_seed(0xc0ffee ^ local_rank)
                routing_topk = _generate_routing_topk(T, top_k, num_experts,
                                                      seed=0xfade ^ local_rank)
                topk_slot_offset = local_rank * top_k

                # GPU
                gpu_gather, gpu_tile, gpu_glayout, gpu_m_t, gpu_psum, gpu_row_topk = \
                    deep_gemm.build_gather_layout_for_rank_overlap(
                        routing_topk, local_rank, num_ranks, tpr, num_experts, block_m,
                        topk_slot_offset=topk_slot_offset,
                        expert_srank_padding=expert_srank_padding)
                torch.cuda.synchronize()
                gpu_m = int(gpu_m_t.item())

                # Reference
                ref_gather, ref_tile, ref_glayout, ref_m, ref_psum, ref_row_topk = reference_build_gather_layout(
                    routing_topk, local_rank, num_ranks, tpr, num_experts, block_m,
                    topk_slot_offset=topk_slot_offset,
                    expert_srank_padding=expert_srank_padding)

                # ---- m_logical ----
                assert gpu_m == ref_m, \
                    (f'm_logical mismatch: gpu={gpu_m}, ref={ref_m} '
                     f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                     f'top_k={top_k}, block_m={block_m}, local_rank={local_rank}, '
                     f'expert_srank_padding={expert_srank_padding})')

                # ---- tile_rank ----
                num_m_tiles = gpu_m // block_m
                gpu_tile_used = gpu_tile[:num_m_tiles]
                assert torch.equal(gpu_tile_used, ref_tile), \
                    (f'tile_rank mismatch (first 16): gpu={gpu_tile_used[:16].tolist()}, '
                     f'ref={ref_tile[:16].tolist()} '
                     f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                     f'local_rank={local_rank}, expert_srank_padding={expert_srank_padding})')

                # ---- grouped_layout ----
                gpu_glayout_used = gpu_glayout[:gpu_m]
                assert torch.equal(gpu_glayout_used, ref_glayout), \
                    (f'grouped_layout mismatch '
                     f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                     f'local_rank={local_rank}, expert_srank_padding={expert_srank_padding})')

                # ---- psum_layout ----
                assert torch.equal(gpu_psum, ref_psum), \
                    (f'psum_layout mismatch: gpu={gpu_psum.tolist()}, ref={ref_psum.tolist()} '
                     f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                     f'local_rank={local_rank}, expert_srank_padding={expert_srank_padding})')

                # ---- gather_index + row_to_topk (per-chunk pair multiset) ----
                counts_cpu = [[0] * num_ranks for _ in range(num_experts)]
                rt_cpu = routing_topk.cpu().numpy()
                for t in range(T):
                    r = t // tpr
                    for j in range(top_k):
                        e = int(rt_cpu[t, j])
                        if 0 <= e < num_experts:
                            counts_cpu[e][r] += 1
                gpu_row_topk_used = gpu_row_topk[:gpu_m]
                ok, msg = _chunk_pair_multiset_equal(
                    gpu_gather[:gpu_m], gpu_row_topk_used, ref_gather, ref_row_topk,
                    counts_cpu, local_rank, num_ranks, num_experts, block_m,
                    expert_srank_padding=expert_srank_padding)
                assert ok, (f'gather_index mismatch: {msg} '
                            f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                            f'local_rank={local_rank}, expert_srank_padding={expert_srank_padding})')
                assert torch.equal(gpu_row_topk_used[gpu_gather[:gpu_m] < 0],
                                   torch.full_like(gpu_row_topk_used[gpu_gather[:gpu_m] < 0], -1))
                real_topk = gpu_row_topk_used[gpu_gather[:gpu_m] >= 0]
                assert ((real_topk >= topk_slot_offset) & (real_topk < topk_slot_offset + top_k)).all(), \
                    (f'row_to_topk out of range '
                     f'(num_ranks={num_ranks}, tpr={tpr}, n_experts={num_experts}, '
                     f'local_rank={local_rank}, expert_srank_padding={expert_srank_padding})')

                n_pad = (gpu_gather[:gpu_m] == -1).sum().item()
                pad_mode = 'expert+srank' if expert_srank_padding else 'expert-only'
                print(f'  > nr={num_ranks}, tpr={tpr:5}, ne={num_experts:3}, k={top_k}, '
                      f'block_m={block_m:3}, lr={local_rank}, pad={pad_mode} '
                      f'| m={gpu_m:6}, n_pad={n_pad:5}  OK')
                summary.append((num_ranks, tpr, num_experts, top_k, block_m,
                                local_rank, pad_mode, gpu_m, n_pad))

    _format_table(
        'Summary: gather-layout generator vs pure-Python reference',
        ['ranks', 'tpr', 'experts', 'top_k', 'block_m', 'local', 'padding', 'm_logical', 'n_pad'],
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
        for expert_srank_padding in (True, False):
            for local_rank in [0, num_ranks - 1]:
                torch.manual_seed(0xfeed ^ local_rank)
                routing_topk = _generate_routing_topk(T, top_k, num_experts,
                                                      seed=0xbeef ^ local_rank)

                # Build the layout
                gather_index, tile_rank, grouped_layout, m_logical_t, psum_layout, row_to_topk = \
                    deep_gemm.build_gather_layout_for_rank_overlap(
                        routing_topk, local_rank, num_ranks, tpr, num_experts, block_m,
                        expert_srank_padding=expert_srank_padding)
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

                gemm_kwargs = {}
                if expert_srank_padding:
                    gemm_kwargs.update(rank_flags=rank_flags,
                                       tile_rank=tile_rank[: m_logical // block_m],
                                       num_ranks=num_ranks,
                                       rank_flag_epoch=1)
                deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                    a_pool_fp8, b_fp8, d, psum_layout,
                    disable_ue8m0_cast=disable_ue8m0_cast,
                    recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                    use_psum_layout=True,
                    expected_m_for_psum_layout=int((m_logical + num_experts - 1) // num_experts * 1.2),
                    gather_index=gather_index[:m_logical],
                    **gemm_kwargs,
                )
                torch.cuda.synchronize()

                # Pad rows must come out exactly 0.
                if is_pad.any():
                    d_pad = d[is_pad]
                    assert torch.all(d_pad == 0), \
                        (f'pad rows non-zero: ranks={num_ranks}, tpr={tpr}, '
                         f'ne={num_experts}, lr={local_rank}, '
                         f'expert_srank_padding={expert_srank_padding}')

                from deep_gemm.testing import calc_diff
                diff = calc_diff(d.to(torch.bfloat16), ref_d)
                max_diff = max(quant_config.max_diff(), 0.02)
                assert diff < max_diff, \
                    (f'e2e diff fails: ranks={num_ranks}, tpr={tpr}, ne={num_experts}, '
                     f'lr={local_rank}, expert_srank_padding={expert_srank_padding}, '
                     f'diff={diff:.5f} >= {max_diff}')

                n_pad = int(is_pad.sum().item())
                pad_mode = 'expert+srank' if expert_srank_padding else 'expert-only'
                print(f'  > nr={num_ranks}, tpr={tpr}, ne={num_experts:3}, k={top_k}, lr={local_rank} '
                      f'pad={pad_mode} | m={m_logical:5}, n_pad={n_pad:5}, diff={diff:.0e}  OK')


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
