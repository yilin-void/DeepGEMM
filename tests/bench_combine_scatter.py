"""Bench grouped GEMM combine-scatter and reduce-scatter combine schemes.

This benchmark intentionally does not time all-gather. Every rank constructs the
same deterministic full A pool so the measured paths focus on fc2 output
movement and combine:

  1. GEMM no scatter
  2. fused GEMM + raw P2P scatter
  3. GEMM + standalone raw scatter-copy
  4. GEMM + pack/local-reduce + NCCL reduce-scatter
  5. fused GEMM + raw scatter + scored source-rank local reduction
"""

import argparse
import os
import socket
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist

import deep_gemm
from deep_gemm.testing import get_arch_major
from deep_gemm.utils.dist import dist_print, init_dist

sys.path.insert(0, os.path.dirname(__file__))
from generators import (  # noqa: E402
    MajorTypeAB,
    QuantConfig,
    cast_fp8_fp4_with_major,
    grouped_cast_fp8_fp4_with_major,
)


def _generate_distinct_routing_topk(num_tokens: int, top_k: int, num_experts: int, seed: int) -> torch.Tensor:
    if top_k > num_experts:
        raise ValueError(f'top_k ({top_k}) must be <= num_experts ({num_experts})')

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    scores = torch.rand((num_tokens, num_experts), dtype=torch.float32,
                        device='cuda', generator=gen)
    return torch.topk(scores, top_k, dim=1).indices.to(torch.int32)


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _make_wgmma_n32_physical_to_logical_index(n: int, device: torch.device) -> torch.Tensor:
    if n % 32 != 0:
        raise ValueError(f'N must be divisible by 32 for WGMMA N permutation, got {n}')
    in_tile_physical = torch.arange(32, device=device)
    pair = in_tile_physical // 8
    lane_group = (in_tile_physical % 8) // 2
    elem = in_tile_physical % 2
    in_tile_logical = lane_group * 8 + pair * 2 + elem
    tile_base = torch.arange(0, n, 32, device=device).unsqueeze(1)
    return (tile_base + in_tile_logical.unsqueeze(0)).reshape(-1).to(torch.long)


def _make_wgmma_n64_physical_to_logical_index(n: int, device: torch.device) -> torch.Tensor:
    if n % 64 != 0:
        raise ValueError(f'N must be divisible by 64 for FP8 WGMMA N permutation, got {n}')
    in_tile_physical = torch.arange(64, device=device)
    half = in_tile_physical // 32
    in_half = in_tile_physical % 32
    pair = in_half // 8
    lane_group = (in_half % 8) // 2
    elem = in_half % 2
    in_tile_logical = lane_group * 16 + half * 8 + pair * 2 + elem
    tile_base = torch.arange(0, n, 64, device=device).unsqueeze(1)
    return (tile_base + in_tile_logical.unsqueeze(0)).reshape(-1).to(torch.long)


def _build_compact_gemm2_layout(routing_topk: torch.Tensor,
                                rank: int,
                                num_ranks: int,
                                tokens_per_rank: int,
                                local_top_k: int,
                                num_experts: int,
                                layout_order: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    alignment = deep_gemm.get_mk_alignment_for_contiguous_layout()
    cursor = 0
    psum_values: list[int] = []
    row_chunks: list[tuple[int, torch.Tensor, torch.Tensor]] = []

    for expert in range(num_experts):
        cursor = _align(cursor, alignment)
        if layout_order == 'token':
            positions = (routing_topk == expert).nonzero(as_tuple=False)
            src_tokens = positions[:, 0].to(torch.int32).contiguous()
            topk_slots = (positions[:, 1].to(torch.int32) + rank * local_top_k).contiguous()
        elif layout_order == 'ring':
            src_chunks = []
            topk_chunks = []
            for step in range(num_ranks):
                src_rank = (rank + step) % num_ranks
                token_start = src_rank * tokens_per_rank
                token_end = token_start + tokens_per_rank
                positions = (routing_topk[token_start:token_end] == expert).nonzero(as_tuple=False)
                if positions.numel() == 0:
                    continue
                src_chunks.append((positions[:, 0].to(torch.int32) + token_start).contiguous())
                topk_chunks.append((positions[:, 1].to(torch.int32) + rank * local_top_k).contiguous())
            if src_chunks:
                src_tokens = torch.cat(src_chunks)
                topk_slots = torch.cat(topk_chunks)
            else:
                src_tokens = torch.empty((0,), dtype=torch.int32, device=routing_topk.device)
                topk_slots = torch.empty((0,), dtype=torch.int32, device=routing_topk.device)
        else:
            raise ValueError(f'unknown compact layout order: {layout_order}')
        row_chunks.append((cursor, src_tokens, topk_slots))
        cursor += int(src_tokens.numel())
        psum_values.append(cursor)

    m_logical = _align(cursor, alignment)
    combine_src_index = torch.full((m_logical,), -1, dtype=torch.int32, device='cuda')
    row_to_topk = torch.full((m_logical,), -1, dtype=torch.int32, device='cuda')
    for start, src_tokens, topk_slots in row_chunks:
        end = start + int(src_tokens.numel())
        if end == start:
            continue
        combine_src_index[start:end] = src_tokens
        row_to_topk[start:end] = topk_slots

    psum_layout = torch.tensor(psum_values, dtype=torch.int32, device='cuda')
    return combine_src_index, psum_layout, row_to_topk, m_logical


def _rank0_print(rank: int, msg: str) -> None:
    if rank == 0:
        print(msg, flush=True)


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(('127.0.0.1', 0))
        return int(sock.getsockname()[1])


def _set_default_nccl_ctas() -> None:
    os.environ.setdefault('NCCL_MIN_CTAS', '64')
    os.environ.setdefault('NCCL_MAX_CTAS', '64')


def _bench_ms(fn, group: dist.ProcessGroup, *, warmups: int, iters: int) -> float:
    times = []
    for _ in range(warmups):
        fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    for _ in range(iters):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        times.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(times)


def _bench_cuda_event_ms(fn, group: dist.ProcessGroup, stream: torch.cuda.Stream, *,
                         warmups: int, iters: int) -> float:
    times = []
    for _ in range(warmups):
        fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    for _ in range(iters):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(stream)
        fn()
        end.record(stream)
        end.synchronize()
        dist.barrier(group=group)
        times.append(start.elapsed_time(end))
    return statistics.median(times)


def _max_across_ranks(value: float, group: dist.ProcessGroup) -> float:
    t = torch.tensor([value], dtype=torch.float64, device='cuda')
    dist.all_reduce(t, op=dist.ReduceOp.MAX, group=group)
    return float(t.item())


def _quantize_cpu_row_per_group(row_value: torch.Tensor, group_n: int) -> torch.Tensor:
    row_value = row_value.float().clone()
    n = int(row_value.numel())
    for col in range(0, n, group_n):
        chunk = row_value[col:col + group_n]
        amax = float(chunk.abs().max().item())
        if amax == 0.0:
            row_value[col:col + group_n] = 0.0
            continue
        scale = amax / 448.0
        row_value[col:col + group_n] = (
            (chunk / scale).to(torch.float8_e4m3fn).float() * scale)
    return row_value


def _select_cpu_reference_rows(combine_src_index: torch.Tensor,
                               row_to_topk: torch.Tensor,
                               tokens_per_rank: int,
                               top_k: int,
                               num_ranks: int,
                               ref_tokens: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
                                                          torch.Tensor, torch.Tensor]:
    gather_cpu = combine_src_index.detach().cpu().to(torch.int64)
    row_topk_cpu = row_to_topk.detach().cpu().to(torch.int64)

    valid = (gather_cpu >= 0) & (row_topk_cpu >= 0) & (row_topk_cpu < top_k)
    valid &= gather_cpu < tokens_per_rank * num_ranks
    rows = torch.nonzero(valid, as_tuple=False).flatten()

    src_tokens = gather_cpu[rows]
    topk_slots = row_topk_cpu[rows]
    src_ranks = torch.div(src_tokens, tokens_per_rank, rounding_mode='floor')
    local_tokens = src_tokens - src_ranks * tokens_per_rank
    in_sample = local_tokens < ref_tokens

    return (rows[in_sample], src_tokens[in_sample], src_ranks[in_sample],
            local_tokens[in_sample], topk_slots[in_sample])


def _build_cpu_combine_reference(d_ref: torch.Tensor,
                                 combine_src_index: torch.Tensor,
                                 row_to_topk: torch.Tensor,
                                 topk_scores: torch.Tensor,
                                 tokens_per_rank: int,
                                 top_k: int,
                                 num_ranks: int,
                                 ref_tokens: int,
                                 group: dist.ProcessGroup,
                                 fp8_scale_group_n: int = 0) -> torch.Tensor:
    d_cpu = d_ref.detach().cpu()
    scores_cpu = topk_scores.detach().cpu()

    n = int(d_cpu.size(1))
    partial = torch.zeros((num_ranks, ref_tokens, n), dtype=torch.float32, device='cpu')
    rows, src_tokens, src_ranks, local_tokens, topk_slots = _select_cpu_reference_rows(
        combine_src_index, row_to_topk, tokens_per_rank, top_k, num_ranks, ref_tokens)

    for row, src_token, src_rank, local_token, topk_slot in zip(
            rows.tolist(), src_tokens.tolist(), src_ranks.tolist(),
            local_tokens.tolist(), topk_slots.tolist()):
        score = float(scores_cpu[src_token, topk_slot])
        row_value = d_cpu[row].float()
        if fp8_scale_group_n > 0:
            row_value = _quantize_cpu_row_per_group(row_value, fp8_scale_group_n)
        partial[src_rank, local_token].add_(row_value, alpha=score)

    partial_gpu = partial.cuda()
    dist.all_reduce(partial_gpu, op=dist.ReduceOp.SUM, group=group)
    return partial_gpu[dist.get_rank(group)].contiguous()


def _build_cpu_slot_reference(d_ref: torch.Tensor,
                              combine_src_index: torch.Tensor,
                              row_to_topk: torch.Tensor,
                              tokens_per_rank: int,
                              top_k: int,
                              num_ranks: int,
                              ref_tokens: int,
                              group: dist.ProcessGroup,
                              fp8_scale_group_n: int = 0) -> torch.Tensor:
    d_cpu = d_ref.detach().cpu()

    n = int(d_cpu.size(1))
    partial = torch.zeros((num_ranks, ref_tokens, top_k, n), dtype=torch.float32, device='cpu')
    rows, _src_tokens, src_ranks, local_tokens, topk_slots = _select_cpu_reference_rows(
        combine_src_index, row_to_topk, tokens_per_rank, top_k, num_ranks, ref_tokens)

    for row, src_rank, local_token, topk_slot in zip(
            rows.tolist(), src_ranks.tolist(), local_tokens.tolist(), topk_slots.tolist()):
        row_value = d_cpu[row].float()
        if fp8_scale_group_n > 0:
            row_value = _quantize_cpu_row_per_group(row_value, fp8_scale_group_n)
        partial[src_rank, local_token, topk_slot].copy_(row_value)

    partial_gpu = partial.cuda()
    dist.all_reduce(partial_gpu, op=dist.ReduceOp.SUM, group=group)
    return partial_gpu[dist.get_rank(group)].contiguous()


def _print_error_stats(label: str,
                       actual: torch.Tensor,
                       reference: torch.Tensor,
                       group: dist.ProcessGroup) -> None:
    diff = (actual - reference).float()
    abs_diff = diff.abs()
    ref_abs = reference.float().abs()
    max_abs_t = abs_diff.max().reshape(1)
    sum_abs_t = abs_diff.sum(dtype=torch.float64).reshape(1)
    sum_sq_t = diff.square().sum(dtype=torch.float64).reshape(1)
    max_ref_t = ref_abs.max().reshape(1)
    count_t = torch.tensor([actual.numel()], dtype=torch.float64, device=actual.device)
    dist.all_reduce(max_abs_t, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(sum_abs_t, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(sum_sq_t, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(max_ref_t, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(count_t, op=dist.ReduceOp.SUM, group=group)
    if dist.get_rank(group) == 0:
        count = float(count_t.item())
        mean_abs = float(sum_abs_t.item() / count)
        rmse = (float(sum_sq_t.item() / count)) ** 0.5
        print(f'  {label}: max_abs={float(max_abs_t.item()):.6g}, '
              f'mean_abs={mean_abs:.6g}, rmse={rmse:.6g}, '
              f'max_ref_abs={float(max_ref_t.item()):.6g}', flush=True)


def _check_against_reference(label: str,
                             actual: torch.Tensor,
                             reference: torch.Tensor,
                             group: dist.ProcessGroup,
                             rtol: float,
                             atol: float) -> tuple[float, int]:
    actual = actual[:reference.size(0), :].contiguous()
    diff = (actual - reference).abs()
    tolerance = atol + rtol * reference.abs()
    nonfinite = ~torch.isfinite(diff)
    finite_diff = torch.nan_to_num(diff, nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
    max_diff_t = finite_diff.max().reshape(1)
    mismatch_count_t = (nonfinite | (finite_diff > tolerance)).sum().reshape(1)
    actual_nonfinite_t = (~torch.isfinite(actual)).sum().reshape(1)
    ref_nonfinite_t = (~torch.isfinite(reference)).sum().reshape(1)
    dist.all_reduce(max_diff_t, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(mismatch_count_t, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(actual_nonfinite_t, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(ref_nonfinite_t, op=dist.ReduceOp.SUM, group=group)
    max_diff = float(max_diff_t.item())
    mismatch_count = int(mismatch_count_t.item())
    actual_nonfinite = int(actual_nonfinite_t.item())
    ref_nonfinite = int(ref_nonfinite_t.item())
    if mismatch_count != 0:
        raise AssertionError(
            f'{label} mismatch vs CPU reference: count={mismatch_count}, '
            f'max_abs_diff={max_diff}, actual_nonfinite={actual_nonfinite}, '
            f'ref_nonfinite={ref_nonfinite}, rtol={rtol}, atol={atol}')
    return max_diff, mismatch_count


def _init_nccl_cpp_comm(rank: int, num_ranks: int, local_rank: int,
                        group: dist.ProcessGroup):
    nccl_unique_ids = [deep_gemm.nccl_get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(nccl_unique_ids, src=0, group=group)
    comm = deep_gemm.nccl_comm_init_rank(nccl_unique_ids[0], rank, num_ranks, local_rank)
    dist.barrier(group=group)
    return comm


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print('SM90 is required for this benchmark.', once_in_node=True)
        return
    if args.top_k != args.experts_per_rank_token * num_ranks:
        raise ValueError('--top-k must equal --experts-per-rank-token * EP/world size')
    if args.global_num_experts % num_ranks != 0:
        raise ValueError('--global-num-experts must be divisible by EP/world size')

    nccl_cpp_comm = _init_nccl_cpp_comm(rank, num_ranks, local_rank, group) if args.bench_reduce_scatter else None

    tokens_per_rank = args.tokens_per_rank
    total_tokens = num_ranks * tokens_per_rank
    hidden = args.hidden
    local_top_k = args.experts_per_rank_token
    combine_top_k = args.top_k
    num_experts = args.global_num_experts // num_ranks
    block_m = args.block_m

    _rank0_print(rank, 'Preparing benchmark tensors...')
    quant_config = QuantConfig()
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    torch.manual_seed(0x2026)
    a_global_bf16 = torch.randn((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')

    routing_topk = _generate_distinct_routing_topk(total_tokens, local_top_k, num_experts, seed=0xBEEF)
    use_compact_gemm2_layout = args.no_gather_a and not args.rank_padded_gemm2_layout
    if use_compact_gemm2_layout:
        _rank0_print(rank, f'Building compact GEMM2 layout ({args.compact_layout_order} order)...')
        combine_src_index, psum_layout, row_to_topk, m_logical = _build_compact_gemm2_layout(
            routing_topk, rank, num_ranks, tokens_per_rank, local_top_k, num_experts,
            args.compact_layout_order)
        gather_index = combine_src_index
    else:
        _rank0_print(rank, 'Building rank-padded gather layout...')
        gather_index, _tile_rank, _grouped_layout, m_logical_t, psum_layout, row_to_topk = \
            deep_gemm.build_gather_layout_for_rank_overlap(
                routing_topk, rank, num_ranks, tokens_per_rank, num_experts, block_m,
                topk_slot_offset=rank * local_top_k)
        m_logical = int(m_logical_t.item())
        combine_src_index = gather_index
    expected_m_per_expert = int((m_logical + num_experts - 1) // num_experts * 1.2)
    real_rows = total_tokens * local_top_k
    _rank0_print(rank, f'GEMM layout ready: m_logical={m_logical}, padding_rows={m_logical - real_rows}')

    if args.no_gather_a:
        _rank0_print(rank, 'Preparing pre-gathered A rows for GEMM2-style no-gather load...')
        gather_used = gather_index[:m_logical].to(torch.int64)
        valid_rows = gather_used >= 0
        a_grouped_bf16 = torch.zeros((m_logical, hidden), dtype=torch.bfloat16, device='cuda')
        a_grouped_bf16[valid_rows] = a_global_bf16[gather_used[valid_rows]]
        a_pool, sfa_global = cast_fp8_fp4_with_major(
            a_grouped_bf16, MajorTypeAB.KMajor, quant_config.gran_k_a,
            quant_config.is_fp4_a, use_ue8m0=False)
        gemm_gather_index = None
    else:
        a_pool, sfa_global = cast_fp8_fp4_with_major(
            a_global_bf16, MajorTypeAB.KMajor, quant_config.gran_k_a,
            quant_config.is_fp4_a, use_ue8m0=False)
        gemm_gather_index = gather_index

    _rank0_print(rank, 'Preparing grouped GEMM weights...')
    n_eff = args.n * args.num_weights
    torch.manual_seed(0x5678 + rank)
    b_bf16 = torch.randn((num_experts, n_eff, hidden), dtype=torch.bfloat16, device='cuda')
    n_physical_to_logical = None
    if args.combine_scatter_direct_accum_stg:
        perm_n = 64 if args.combine_scatter_fp8 else 32
        _rank0_print(rank, f'Permuting B within each {perm_n}-wide N chunk for direct accumulator stores...')
        n_physical_to_logical = (
            _make_wgmma_n64_physical_to_logical_index(n_eff, b_bf16.device)
            if args.combine_scatter_fp8
            else _make_wgmma_n32_physical_to_logical_index(n_eff, b_bf16.device)
        )
        b_bf16 = b_bf16.index_select(1, n_physical_to_logical).contiguous()
    b_fp8 = grouped_cast_fp8_fp4_with_major(
        b_bf16, MajorTypeAB.KMajor, quant_config.gran_k_b,
        quant_config.is_fp4_b, use_ue8m0=False, use_block_cast_for_fp8=True)

    d = torch.empty((m_logical, n_eff), dtype=torch.bfloat16, device='cuda')
    peer_tma_ptr_override = None
    peer_tma_dst_rank = None
    if args.bench_peer_tma_store or args.bench_peer_stg_store:
        peer_tma_d_local = torch.empty_like(d)
        peer_tma_handles = [None] * num_ranks
        dist.all_gather_object(peer_tma_handles, deep_gemm.cuda_ipc_get_mem_handle(peer_tma_d_local), group=group)
        peer_tma_ptrs = deep_gemm.cuda_ipc_open_mem_handles(peer_tma_handles, rank, peer_tma_d_local)
        peer_tma_dst_rank = (rank + 1) % num_ranks
        peer_tma_ptr_override = int(peer_tma_ptrs[peer_tma_dst_rank])

    torch.manual_seed(0x3456)
    combine_topk_scores = torch.rand((total_tokens, combine_top_k), dtype=torch.float32, device='cuda')
    local_topk_scores = combine_topk_scores[
        rank * tokens_per_rank:(rank + 1) * tokens_per_rank, :combine_top_k].contiguous()

    combine_dtype = torch.float8_e4m3fn if args.combine_scatter_fp8 else torch.bfloat16
    combine_buffer = torch.empty((tokens_per_rank, combine_top_k, n_eff),
                                 dtype=combine_dtype, device='cuda')
    combine_scales = None
    combine_scale_ptrs_t = None
    if args.combine_scatter_fp8:
        combine_scales = torch.empty((tokens_per_rank, combine_top_k, n_eff // 32),
                                     dtype=torch.float32, device='cuda')
    combine_handles = [None] * num_ranks
    dist.all_gather_object(combine_handles, deep_gemm.cuda_ipc_get_mem_handle(combine_buffer), group=group)
    combine_buffer_ptrs = deep_gemm.cuda_ipc_open_mem_handles(combine_handles, rank, combine_buffer)
    if args.combine_scatter_fp8:
        combine_scale_handles = [None] * num_ranks
        dist.all_gather_object(combine_scale_handles, deep_gemm.cuda_ipc_get_mem_handle(combine_scales), group=group)
        combine_scale_ptrs = deep_gemm.cuda_ipc_open_mem_handles(combine_scale_handles, rank, combine_scales)
    if args.combine_scatter_local_buffer:
        _rank0_print(rank, 'Redirecting combine-scatter stores to a local mirror buffer...')
        local_scatter_buffer = torch.empty((num_ranks, tokens_per_rank, combine_top_k, n_eff),
                                           dtype=combine_dtype, device='cuda')
        base_ptr = int(local_scatter_buffer.data_ptr())
        rank_stride_bytes = local_scatter_buffer.stride(0) * local_scatter_buffer.element_size()
        combine_buffer_ptrs = [base_ptr + src_rank * rank_stride_bytes for src_rank in range(num_ranks)]
        if args.combine_scatter_fp8:
            local_scatter_scales = torch.empty((num_ranks, tokens_per_rank, combine_top_k, n_eff // 32),
                                               dtype=torch.float32, device='cuda')
            scale_base_ptr = int(local_scatter_scales.data_ptr())
            scale_rank_stride_bytes = local_scatter_scales.stride(0) * local_scatter_scales.element_size()
            combine_scale_ptrs = [
                scale_base_ptr + src_rank * scale_rank_stride_bytes for src_rank in range(num_ranks)]
    combine_buffer_ptrs_t = torch.tensor(combine_buffer_ptrs, dtype=torch.int64, device='cuda')
    if args.combine_scatter_fp8:
        combine_scale_ptrs_t = torch.tensor(combine_scale_ptrs, dtype=torch.int64, device='cuda')

    reduce_scatter_input = None
    reduce_scatter_output = None
    if args.bench_reduce_scatter:
        reduce_scatter_input = torch.empty((num_ranks, tokens_per_rank, n_eff),
                                           dtype=torch.float32, device='cuda')
        reduce_scatter_output = torch.empty((tokens_per_rank, n_eff), dtype=torch.float32, device='cuda')
    fused_reduce_output = torch.empty((tokens_per_rank, n_eff), dtype=torch.float32, device='cuda')

    compute_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    def launch_gemm(enable_combine_scatter: bool,
                    use_tma_store: bool = True,
                    tma_store_ptr_override: int | None = None) -> None:
        kw = {}
        if enable_combine_scatter:
            kw.update(combine_src_index=combine_src_index,
                      combine_row_topk=row_to_topk,
                      combine_buffer_ptrs=combine_buffer_ptrs_t,
                      combine_tokens_per_rank=tokens_per_rank,
                      combine_top_k=combine_top_k,
                      combine_scatter_direct_accum_stg=args.combine_scatter_direct_accum_stg,
                      combine_scatter_fp8=args.combine_scatter_fp8)
            if args.combine_scatter_fp8:
                kw.update(combine_scale_ptrs=combine_scale_ptrs_t)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_pool, sfa_global), b_fp8, d, psum_layout,
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
            disable_ue8m0_cast=True,
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m_per_expert,
            gather_index=gemm_gather_index,
            use_tma_store=False if enable_combine_scatter else use_tma_store,
            tma_store_ptr_override=tma_store_ptr_override,
            **kw,
        )

    def run_gemm_no_scatter() -> None:
        with torch.cuda.stream(compute_stream):
            launch_gemm(enable_combine_scatter=False)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_fused_scatter() -> None:
        with torch.cuda.stream(compute_stream):
            launch_gemm(enable_combine_scatter=True)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_peer_tma_store() -> None:
        with torch.cuda.stream(compute_stream):
            launch_gemm(enable_combine_scatter=False,
                        use_tma_store=True,
                        tma_store_ptr_override=peer_tma_ptr_override)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_peer_stg_store() -> None:
        with torch.cuda.stream(compute_stream):
            launch_gemm(enable_combine_scatter=False,
                        use_tma_store=False,
                        tma_store_ptr_override=peer_tma_ptr_override)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_standalone_scatter() -> None:
        with torch.cuda.stream(compute_stream):
            deep_gemm.combine_scatter_copy_rows(
                d, combine_src_index, row_to_topk, combine_buffer_ptrs_t,
                tokens_per_rank, combine_top_k, args.scatter_rows_per_block)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_two_stage_scatter() -> None:
        with torch.cuda.stream(compute_stream):
            launch_gemm(enable_combine_scatter=False)
            deep_gemm.combine_scatter_copy_rows(
                d, combine_src_index, row_to_topk, combine_buffer_ptrs_t,
                tokens_per_rank, combine_top_k, args.scatter_rows_per_block)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_pack_for_reduce_scatter() -> None:
        with torch.cuda.stream(compute_stream):
            reduce_scatter_input.zero_()
            deep_gemm.combine_pack_for_reduce_scatter(
                d, combine_src_index, row_to_topk, combine_topk_scores, reduce_scatter_input,
                tokens_per_rank, combine_top_k)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_nccl_reduce_scatter() -> None:
        with torch.cuda.stream(comm_stream):
            deep_gemm.nccl_reduce_scatter_sum(reduce_scatter_input, reduce_scatter_output, nccl_cpp_comm)
        torch.cuda.current_stream().wait_stream(comm_stream)

    def run_reduce_scatter_baseline() -> None:
        with torch.cuda.stream(compute_stream):
            launch_gemm(enable_combine_scatter=False)
            reduce_scatter_input.zero_()
            deep_gemm.combine_pack_for_reduce_scatter(
                d, combine_src_index, row_to_topk, combine_topk_scores, reduce_scatter_input,
                tokens_per_rank, combine_top_k)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_stream(compute_stream)
            deep_gemm.nccl_reduce_scatter_sum(reduce_scatter_input, reduce_scatter_output, nccl_cpp_comm)
        torch.cuda.current_stream().wait_stream(compute_stream)
        torch.cuda.current_stream().wait_stream(comm_stream)

    def run_local_reduce_only() -> None:
        with torch.cuda.stream(compute_stream):
            if args.combine_scatter_fp8:
                deep_gemm.combine_reduce_slots_fp8(
                    combine_buffer, combine_scales, local_topk_scores, fused_reduce_output)
            else:
                deep_gemm.combine_reduce_slots(combine_buffer, local_topk_scores, fused_reduce_output)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_fused_scatter_then_local_reduce() -> None:
        run_fused_scatter()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        run_local_reduce_only()

    _rank0_print(rank, 'Warming JIT paths...')
    run_gemm_no_scatter()
    run_fused_scatter()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    if args.check:
        _rank0_print(rank, 'Checking combine-scatter values against non-scatter GEMM...')
        run_gemm_no_scatter()
        torch.cuda.synchronize()
        dist.barrier(group=group)

        def check_scatter_output(label: str, fn) -> None:
            if not args.combine_scatter_fp8:
                combine_buffer.fill_(float('nan'))
            torch.cuda.synchronize()
            dist.barrier(group=group)
            fn()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            d_for_check = d
            if n_physical_to_logical is not None:
                d_for_check = torch.empty_like(d)
                d_for_check[:, n_physical_to_logical] = d
            max_diff_t, mismatch_count_t = deep_gemm.check_combine_scatter_output(
                d_for_check, combine_src_index, row_to_topk, combine_buffer_ptrs_t,
                tokens_per_rank, combine_top_k, 0.0)
            torch.cuda.synchronize()
            dist.all_reduce(max_diff_t, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(mismatch_count_t, op=dist.ReduceOp.SUM, group=group)
            max_diff = float(max_diff_t.item())
            mismatch_count = int(mismatch_count_t.item())
            if mismatch_count != 0:
                raise AssertionError(
                    f'{label} mismatch: count={mismatch_count}, max_abs_diff={max_diff}')
            _rank0_print(rank, f'{label} value check passed: max_abs_diff=0, mismatches=0')

        if args.combine_scatter_fp8:
            _rank0_print(rank, 'Skipping exact raw scatter check for FP8 combine-scatter.')
        else:
            check_scatter_output('Fused combine-scatter epilogue', run_fused_scatter)
            if args.bench_standalone_scatter:
                check_scatter_output('Standalone scatter-copy', run_standalone_scatter)

        cpu_ref_tokens = tokens_per_rank if args.cpu_ref_tokens < 0 else min(args.cpu_ref_tokens, tokens_per_rank)
        cpu_reference = None
        if cpu_ref_tokens > 0:
            _rank0_print(rank, f'Building CPU combine reference for first {cpu_ref_tokens} local tokens...')
            run_gemm_no_scatter()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            d_for_cpu_ref = d
            if n_physical_to_logical is not None:
                d_for_cpu_ref = torch.empty_like(d)
                d_for_cpu_ref[:, n_physical_to_logical] = d
            cpu_reference = _build_cpu_combine_reference(
                d_for_cpu_ref, combine_src_index, row_to_topk, combine_topk_scores,
                tokens_per_rank, combine_top_k, num_ranks, cpu_ref_tokens, group,
                fp8_scale_group_n=32 if args.combine_scatter_fp8 else 0)
            torch.cuda.synchronize()
            dist.barrier(group=group)

        if cpu_reference is not None:
            _rank0_print(rank, 'Checking fused scatter + local reduction against CPU reference...')
            run_fused_scatter()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            if not args.combine_scatter_fp8:
                combine_nonfinite_count_t = (~torch.isfinite(combine_buffer)).sum().reshape(1)
                dist.all_reduce(combine_nonfinite_count_t, op=dist.ReduceOp.SUM, group=group)
                combine_nonfinite_count = int(combine_nonfinite_count_t.item())
                if combine_nonfinite_count != 0:
                    raise AssertionError(f'Fused combine buffer has non-finite values: count={combine_nonfinite_count}')
            run_local_reduce_only()
            torch.cuda.synchronize()
            dist.barrier(group=group)

            if args.combine_scatter_fp8 and args.fp8_debug_precision:
                _rank0_print(rank, 'FP8 precision diagnostics:')
                cpu_unquant_reference = _build_cpu_combine_reference(
                    d_for_cpu_ref, combine_src_index, row_to_topk, combine_topk_scores,
                    tokens_per_rank, combine_top_k, num_ranks, cpu_ref_tokens, group)
                cpu_quant_slots = _build_cpu_slot_reference(
                    d_for_cpu_ref, combine_src_index, row_to_topk,
                    tokens_per_rank, combine_top_k, num_ranks, cpu_ref_tokens, group,
                    fp8_scale_group_n=32)
                cpu_bf16_slots = _build_cpu_slot_reference(
                    d_for_cpu_ref, combine_src_index, row_to_topk,
                    tokens_per_rank, combine_top_k, num_ranks, cpu_ref_tokens, group)
                actual_slots = combine_buffer[:cpu_ref_tokens].float()
                actual_scales = combine_scales[:cpu_ref_tokens].repeat_interleave(32, dim=2)
                actual_dequant_slots = actual_slots * actual_scales
                slot_reduce_reference = (
                    actual_dequant_slots * local_topk_scores[:cpu_ref_tokens, :, None]).sum(dim=1)
                _print_error_stats('local-reduce-kernel vs torch-reduce(actual slots)',
                                   fused_reduce_output[:cpu_ref_tokens], slot_reduce_reference, group)
                _print_error_stats('actual dequant slots vs CPU quantized slots',
                                   actual_dequant_slots, cpu_quant_slots, group)
                _print_error_stats('CPU quantized slots vs CPU BF16 slots',
                                   cpu_quant_slots, cpu_bf16_slots, group)
                _print_error_stats('actual dequant slots vs CPU BF16 slots',
                                   actual_dequant_slots, cpu_bf16_slots, group)
                _print_error_stats('actual reduction vs CPU quantized reference',
                                   fused_reduce_output[:cpu_ref_tokens], cpu_reference, group)
                _print_error_stats('actual reduction vs CPU BF16 reference',
                                   fused_reduce_output[:cpu_ref_tokens], cpu_unquant_reference, group)

            max_diff, _ = _check_against_reference(
                'Fused scatter + local reduction', fused_reduce_output, cpu_reference, group,
                args.reduce_check_rtol, args.reduce_check_atol)
            _rank0_print(rank, f'CPU reference check passed for fused scatter + local reduction: '
                         f'max_abs_diff={max_diff:.6g}, tokens={cpu_ref_tokens}, mismatches=0')

        if args.bench_reduce_scatter:
            _rank0_print(rank, 'Checking reduce-scatter baseline against fused scatter + local reduction...')
            run_reduce_scatter_baseline()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            run_fused_scatter()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            combine_nonfinite_count_t = (~torch.isfinite(combine_buffer)).sum().reshape(1)
            dist.all_reduce(combine_nonfinite_count_t, op=dist.ReduceOp.SUM, group=group)
            combine_nonfinite_count = int(combine_nonfinite_count_t.item())
            if combine_nonfinite_count != 0:
                raise AssertionError(f'Fused combine buffer has non-finite values: count={combine_nonfinite_count}')
            run_local_reduce_only()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            diff = (fused_reduce_output - reduce_scatter_output).abs()
            tolerance = args.reduce_check_atol + args.reduce_check_rtol * reduce_scatter_output.abs()
            nonfinite = ~torch.isfinite(diff)
            finite_diff = torch.nan_to_num(diff, nan=float('inf'), posinf=float('inf'), neginf=float('inf'))
            max_diff_t = finite_diff.max().reshape(1)
            mismatch_count_t = (nonfinite | (finite_diff > tolerance)).sum().reshape(1)
            nonfinite_diff_count_t = nonfinite.sum().reshape(1)
            fused_nonfinite_count_t = (~torch.isfinite(fused_reduce_output)).sum().reshape(1)
            baseline_nonfinite_count_t = (~torch.isfinite(reduce_scatter_output)).sum().reshape(1)
            dist.all_reduce(max_diff_t, op=dist.ReduceOp.MAX, group=group)
            dist.all_reduce(mismatch_count_t, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(nonfinite_diff_count_t, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(fused_nonfinite_count_t, op=dist.ReduceOp.SUM, group=group)
            dist.all_reduce(baseline_nonfinite_count_t, op=dist.ReduceOp.SUM, group=group)
            max_diff = float(max_diff_t.item())
            mismatch_count = int(mismatch_count_t.item())
            nonfinite_diff_count = int(nonfinite_diff_count_t.item())
            fused_nonfinite_count = int(fused_nonfinite_count_t.item())
            baseline_nonfinite_count = int(baseline_nonfinite_count_t.item())
            if mismatch_count != 0:
                raise AssertionError(
                    f'Reduce-scatter baseline mismatch: count={mismatch_count}, max_abs_diff={max_diff}, '
                    f'nonfinite_diff={nonfinite_diff_count}, '
                    f'fused_nonfinite={fused_nonfinite_count}, '
                    f'baseline_nonfinite={baseline_nonfinite_count}, '
                    f'rtol={args.reduce_check_rtol}, atol={args.reduce_check_atol}')
            _rank0_print(rank, f'Reduce-scatter baseline check passed: max_abs_diff={max_diff:.6g}, '
                         f'mismatches=0')
            if cpu_reference is not None:
                max_diff, _ = _check_against_reference(
                    'Reduce-scatter baseline', reduce_scatter_output, cpu_reference, group,
                    args.reduce_check_rtol, args.reduce_check_atol)
                _rank0_print(rank, f'CPU reference check passed for reduce-scatter baseline: '
                             f'max_abs_diff={max_diff:.6g}, tokens={cpu_ref_tokens}, mismatches=0')

    _rank0_print(rank, 'Benchmarking GEMM without combine-scatter...')
    gemm_no_scatter_ms = _max_across_ranks(
        _bench_ms(run_gemm_no_scatter, group, warmups=args.warmups, iters=args.iters), group)
    gemm_no_scatter_event_ms = _max_across_ranks(
        _bench_cuda_event_ms(run_gemm_no_scatter, group, compute_stream,
                             warmups=args.warmups, iters=args.iters), group)

    _rank0_print(rank, 'Benchmarking fused combine-scatter GEMM...')
    fused_scatter_ms = _max_across_ranks(
        _bench_ms(run_fused_scatter, group, warmups=args.warmups, iters=args.iters), group)
    fused_scatter_event_ms = _max_across_ranks(
        _bench_cuda_event_ms(run_fused_scatter, group, compute_stream,
                             warmups=args.warmups, iters=args.iters), group)

    peer_tma_store_ms = None
    peer_tma_store_event_ms = None
    if args.bench_peer_tma_store:
        _rank0_print(rank, 'Benchmarking row-major peer TMA store GEMM...')
        peer_tma_store_ms = _max_across_ranks(
            _bench_ms(run_peer_tma_store, group, warmups=args.warmups, iters=args.iters), group)
        peer_tma_store_event_ms = _max_across_ranks(
            _bench_cuda_event_ms(run_peer_tma_store, group, compute_stream,
                                 warmups=args.warmups, iters=args.iters), group)
    peer_stg_store_ms = None
    peer_stg_store_event_ms = None
    if args.bench_peer_stg_store:
        _rank0_print(rank, 'Benchmarking row-major peer STG store GEMM...')
        peer_stg_store_ms = _max_across_ranks(
            _bench_ms(run_peer_stg_store, group, warmups=args.warmups, iters=args.iters), group)
        peer_stg_store_event_ms = _max_across_ranks(
            _bench_cuda_event_ms(run_peer_stg_store, group, compute_stream,
                                 warmups=args.warmups, iters=args.iters), group)

    standalone_scatter_ms = None
    standalone_scatter_event_ms = None
    two_stage_scatter_ms = None
    two_stage_scatter_event_ms = None
    if args.bench_standalone_scatter:
        _rank0_print(rank, 'Benchmarking standalone scatter-copy...')
        run_gemm_no_scatter()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        standalone_scatter_ms = _max_across_ranks(
            _bench_ms(run_standalone_scatter, group, warmups=args.warmups, iters=args.iters), group)
        standalone_scatter_event_ms = _max_across_ranks(
            _bench_cuda_event_ms(run_standalone_scatter, group, compute_stream,
                                 warmups=args.warmups, iters=args.iters), group)
        _rank0_print(rank, 'Benchmarking two-stage GEMM + scatter-copy...')
        two_stage_scatter_ms = _max_across_ranks(
            _bench_ms(run_two_stage_scatter, group, warmups=args.warmups, iters=args.iters), group)
        two_stage_scatter_event_ms = _max_across_ranks(
            _bench_cuda_event_ms(run_two_stage_scatter, group, compute_stream,
                                 warmups=args.warmups, iters=args.iters), group)

    pack_reduce_ms = None
    pack_reduce_event_ms = None
    reduce_scatter_ms = None
    reduce_scatter_event_ms = None
    reduce_scatter_baseline_ms = None
    if args.bench_reduce_scatter:
        _rank0_print(rank, 'Benchmarking pack/local-reduce for reduce-scatter...')
        run_gemm_no_scatter()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        pack_reduce_ms = _max_across_ranks(
            _bench_ms(run_pack_for_reduce_scatter, group, warmups=args.warmups, iters=args.iters), group)
        pack_reduce_event_ms = _max_across_ranks(
            _bench_cuda_event_ms(run_pack_for_reduce_scatter, group, compute_stream,
                                 warmups=args.warmups, iters=args.iters), group)
        run_pack_for_reduce_scatter()
        torch.cuda.synchronize()
        dist.barrier(group=group)
        _rank0_print(rank, 'Benchmarking NCCL reduce-scatter...')
        reduce_scatter_ms = _max_across_ranks(
            _bench_ms(run_nccl_reduce_scatter, group, warmups=args.warmups, iters=args.iters), group)
        reduce_scatter_event_ms = _max_across_ranks(
            _bench_cuda_event_ms(run_nccl_reduce_scatter, group, comm_stream,
                                 warmups=args.warmups, iters=args.iters), group)
        _rank0_print(rank, 'Benchmarking GEMM + pack/local-reduce + NCCL reduce-scatter baseline...')
        reduce_scatter_baseline_ms = _max_across_ranks(
            _bench_ms(run_reduce_scatter_baseline, group, warmups=args.warmups, iters=args.iters), group)

    _rank0_print(rank, 'Benchmarking local reduction after fused combine-scatter...')
    run_fused_scatter()
    torch.cuda.synchronize()
    dist.barrier(group=group)
    local_reduce_ms = _max_across_ranks(
        _bench_ms(run_local_reduce_only, group, warmups=args.warmups, iters=args.iters), group)
    local_reduce_event_ms = _max_across_ranks(
        _bench_cuda_event_ms(run_local_reduce_only, group, compute_stream,
                             warmups=args.warmups, iters=args.iters), group)
    _rank0_print(rank, 'Benchmarking fused combine-scatter + local reduction...')
    fused_scatter_reduce_ms = _max_across_ranks(
        _bench_ms(run_fused_scatter_then_local_reduce, group, warmups=args.warmups, iters=args.iters), group)

    if rank == 0:
        print('Grouped GEMM combine-scatter bench:', flush=True)
        print(f'  ranks={num_ranks}, tokens/rank={tokens_per_rank}, hidden={hidden}', flush=True)
        print(f'  global_experts={args.global_num_experts}, local_experts={num_experts}, '
              f'global_top_k={combine_top_k}, local_top_k={local_top_k}', flush=True)
        print(f'  m_logical={m_logical}, n={args.n}, num_weights={args.num_weights}, n_eff={n_eff}', flush=True)
        layout_name = f'compact-gemm2/{args.compact_layout_order}' if use_compact_gemm2_layout else 'rank-padded'
        print(f'  gather_a={not args.no_gather_a}, layout={layout_name}, '
              f'direct_accum_stg={args.combine_scatter_direct_accum_stg}, '
              f'fp8_scatter={args.combine_scatter_fp8}', flush=True)
        if args.bench_reduce_scatter:
            print(f'  nccl_ctas={os.environ["NCCL_MIN_CTAS"]}/{os.environ["NCCL_MAX_CTAS"]}', flush=True)
        print('  common components:', flush=True)
        print(f'    GEMM no scatter       : {gemm_no_scatter_ms * 1e3:8.2f} us '
              f'(event {gemm_no_scatter_event_ms * 1e3:8.2f} us)', flush=True)
        print('  fused scatter scheme:', flush=True)
        print(f'    fused GEMM+scatter    : {fused_scatter_ms * 1e3:8.2f} us '
              f'(event {fused_scatter_event_ms * 1e3:8.2f} us)', flush=True)
        if peer_tma_store_ms is not None:
            print('  row-major peer TMA experiment:', flush=True)
            print(f'    peer dst rank offset  : {(peer_tma_dst_rank - rank) % num_ranks}', flush=True)
            print(f'    GEMM peer TMA store   : {peer_tma_store_ms * 1e3:8.2f} us '
                  f'(event {peer_tma_store_event_ms * 1e3:8.2f} us)', flush=True)
        if peer_stg_store_ms is not None:
            print('  row-major peer STG experiment:', flush=True)
            print(f'    peer dst rank offset  : {(peer_tma_dst_rank - rank) % num_ranks}', flush=True)
            print(f'    GEMM peer STG store   : {peer_stg_store_ms * 1e3:8.2f} us '
                  f'(event {peer_stg_store_event_ms * 1e3:8.2f} us)', flush=True)
        if standalone_scatter_ms is not None:
            scatter_bytes = tokens_per_rank * combine_top_k * n_eff * d.element_size()
            scatter_bw = scatter_bytes / (standalone_scatter_event_ms / 1e3) / 1e9
            print('  two-stage scatter-copy scheme:', flush=True)
            print(f'    GEMM no scatter       : {gemm_no_scatter_ms * 1e3:8.2f} us '
                  f'(event {gemm_no_scatter_event_ms * 1e3:8.2f} us)', flush=True)
            print(f'    scatter-copy          : {standalone_scatter_ms * 1e3:8.2f} us '
                  f'(event {standalone_scatter_event_ms * 1e3:8.2f} us, '
                  f'{scatter_bw:7.2f} GB/s)', flush=True)
            print(f'    total GEMM+scatter    : {two_stage_scatter_ms * 1e3:8.2f} us '
                  f'(event {two_stage_scatter_event_ms * 1e3:8.2f} us)', flush=True)
        if pack_reduce_ms is not None:
            rs_bytes = tokens_per_rank * n_eff * reduce_scatter_output.element_size()
            rs_bw = rs_bytes * (num_ranks - 1) / (reduce_scatter_event_ms / 1e3) / 1e9
            print('  reduce-scatter baseline scheme:', flush=True)
            print(f'    GEMM no scatter       : {gemm_no_scatter_ms * 1e3:8.2f} us '
                  f'(event {gemm_no_scatter_event_ms * 1e3:8.2f} us)', flush=True)
            print(f'    pack/local-reduce     : {pack_reduce_ms * 1e3:8.2f} us '
                  f'(event {pack_reduce_event_ms * 1e3:8.2f} us)', flush=True)
            print(f'    NCCL reduce-scatter   : {reduce_scatter_ms * 1e3:8.2f} us '
                  f'(event {reduce_scatter_event_ms * 1e3:8.2f} us, '
                  f'alg-bw {rs_bw:7.2f} GB/s)', flush=True)
            print(f'    total GEMM+pack+RS    : {reduce_scatter_baseline_ms * 1e3:8.2f} us', flush=True)
        print('  fused scatter + local reduction scheme:', flush=True)
        print(f'    fused GEMM+scatter    : {fused_scatter_ms * 1e3:8.2f} us '
              f'(event {fused_scatter_event_ms * 1e3:8.2f} us)', flush=True)
        print(f'    local reduce          : {local_reduce_ms * 1e3:8.2f} us '
              f'(event {local_reduce_event_ms * 1e3:8.2f} us)', flush=True)
        print(f'    total fused+reduce    : {fused_scatter_reduce_ms * 1e3:8.2f} us', flush=True)

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-local-ranks', type=int, default=None)
    parser.add_argument('--tokens-per-rank', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--top-k', type=int, default=16,
                        help='Global top-k across all EP ranks')
    parser.add_argument('--global-num-experts', type=int, default=512)
    parser.add_argument('--experts-per-rank-token', type=int, default=2,
                        help='Number of local experts selected per token on each EP rank')
    parser.add_argument('--no-gather-a', action='store_true',
                        help='Use pre-gathered contiguous A/SFA rows while keeping combine_src_index for scatter')
    parser.add_argument('--rank-padded-gemm2-layout', action='store_true',
                        help='With --no-gather-a, keep the old rank-padded overlap layout instead of compact '
                             'expert-only GEMM2 layout')
    parser.add_argument('--compact-layout-order', choices=('token', 'ring'), default='token',
                        help='Row order for compact GEMM2 layout. token keeps global-token order; ring visits '
                             'source ranks in local ring order inside each expert.')
    parser.add_argument('--bench-standalone-scatter', action='store_true',
                        help='Benchmark separate D -> peer combine-buffer scatter-copy and two-stage total')
    parser.add_argument('--bench-peer-tma-store', action='store_true',
                        help='Benchmark a row-major no-scatter GEMM whose TMA epilogue writes D to a peer rank')
    parser.add_argument('--bench-peer-stg-store', action='store_true',
                        help='Benchmark a row-major no-scatter GEMM whose STG epilogue writes D to a peer rank')
    parser.add_argument('--combine-scatter-direct-accum-stg', action='store_true',
                        help='Use the direct-accumulator combine-scatter epilogue '
                             '(BF16 uses N32 B permutation; FP8 uses N64)')
    parser.add_argument('--combine-scatter-fp8', action='store_true',
                        help='Use E4M3 FP8 combine-scatter data with FP32 scales per row per 32 columns')
    parser.add_argument('--combine-scatter-local-buffer', action='store_true',
                        help='Benchmark-only: redirect combine-scatter destination pointers to local HBM slices '
                             'instead of peer IPC buffers')
    parser.add_argument('--bench-reduce-scatter', action='store_true',
                        help='Benchmark GEMM + local pack/reduce + NCCL reduce-scatter baseline')
    parser.add_argument('--scatter-rows-per-block', type=int, default=4,
                        choices=(1, 2, 4, 8, 16, 32),
                        help='Rows handled by each standalone scatter-copy CTA')
    parser.add_argument('--reduce-check-rtol', type=float, default=1e-2,
                        help='Relative tolerance for fused scatter+local-reduce vs reduce-scatter baseline')
    parser.add_argument('--reduce-check-atol', type=float, default=2e-2,
                        help='Absolute tolerance for fused scatter+local-reduce vs reduce-scatter baseline')
    parser.add_argument('--cpu-ref-tokens', type=int, default=32,
                        help='When --check is set, compare against a CPU arithmetic reference for this many '
                             'local tokens per rank. Use -1 for all local tokens, or 0 to disable.')
    parser.add_argument('--num-weights', type=int, default=1,
                        help='Number of weight matrices per expert (e.g. 2 for fused gate+up)')
    parser.add_argument('--block-m', type=int, default=128)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--fp8-debug-precision', action='store_true',
                        help='With --check and --combine-scatter-fp8, print slot-level quantization diagnostics')
    args = parser.parse_args()
    if args.cpu_ref_tokens < -1:
        raise ValueError('--cpu-ref-tokens must be -1, 0, or a positive integer')
    if args.combine_scatter_fp8 and not args.combine_scatter_direct_accum_stg:
        raise ValueError('--combine-scatter-fp8 requires --combine-scatter-direct-accum-stg')
    if args.combine_scatter_fp8 and args.n % 64 != 0:
        raise ValueError('--combine-scatter-fp8 requires --n divisible by 64')
    if args.fp8_debug_precision and (not args.check or not args.combine_scatter_fp8):
        raise ValueError('--fp8-debug-precision requires --check and --combine-scatter-fp8')

    if args.bench_reduce_scatter:
        _set_default_nccl_ctas()

    num_local_ranks = args.num_local_ranks or torch.cuda.device_count()
    assert num_local_ranks > 0
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(_find_free_port())
        print(f'Using MASTER_PORT={os.environ["MASTER_PORT"]}', flush=True)
    torch.multiprocessing.spawn(_worker, args=(num_local_ranks, args), nprocs=num_local_ranks)


if __name__ == '__main__':
    main()
