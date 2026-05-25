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
    a_pool, sfa_global = cast_fp8_fp4_with_major(
        a_global_bf16, MajorTypeAB.KMajor, quant_config.gran_k_a,
        quant_config.is_fp4_a, use_ue8m0=False)

    routing_topk = _generate_distinct_routing_topk(total_tokens, combine_top_k, args.global_num_experts, seed=0xBEEF)
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
            deep_gemm_moe_L2.build_gather_layout_for_rank_overlap(
                routing_topk, rank, num_ranks, tokens_per_rank, num_experts, block_m,
                topk_slot_offset=0)
        m_logical = int(m_logical_t.item())
        combine_src_index = gather_index
    expected_m_per_expert = int((m_logical + num_experts - 1) // num_experts * 1.2)
    _rank0_print(rank, f'Gather layout ready: m_logical={m_logical}')

    _rank0_print(rank, 'Preparing grouped GEMM weights...')
    n_eff = args.n * args.num_weights
    torch.manual_seed(0x5678 + rank)
    b_bf16 = torch.randn((num_experts, n_eff, hidden), dtype=torch.bfloat16, device='cuda')
    b_fp8 = grouped_cast_fp8_fp4_with_major(
        b_bf16, MajorTypeAB.KMajor, quant_config.gran_k_b,
        quant_config.is_fp4_b, use_ue8m0=False, use_block_cast_for_fp8=True)

    d = torch.empty((m_logical, n_eff), dtype=torch.bfloat16, device='cuda')
    torch.manual_seed(0x3456)
    combine_topk_scores = torch.rand((total_tokens, combine_top_k), dtype=torch.float32, device='cuda')
    local_topk_scores = combine_topk_scores[
        rank * tokens_per_rank:(rank + 1) * tokens_per_rank, :combine_top_k].contiguous()

    combine_buffer = torch.empty((tokens_per_rank, combine_top_k, n_eff),
                                 dtype=torch.bfloat16, device='cuda')
    combine_handles = [None] * num_ranks
    dist.all_gather_object(combine_handles, deep_gemm.cuda_ipc_get_mem_handle(combine_buffer), group=group)
    combine_buffer_ptrs = deep_gemm.cuda_ipc_open_mem_handles(combine_handles, rank, combine_buffer)
    combine_buffer_ptrs_t = torch.tensor(combine_buffer_ptrs, dtype=torch.int64, device='cuda')

    reduce_scatter_input = None
    reduce_scatter_output = None
    if args.bench_reduce_scatter:
        reduce_scatter_input = torch.empty((num_ranks, tokens_per_rank, n_eff),
                                           dtype=torch.float32, device='cuda')
        reduce_scatter_output = torch.empty((tokens_per_rank, n_eff), dtype=torch.float32, device='cuda')
    fused_reduce_output = torch.empty((tokens_per_rank, n_eff), dtype=torch.float32, device='cuda')

    compute_stream = torch.cuda.Stream()
    comm_stream = torch.cuda.Stream()

    def launch_gemm(enable_combine_scatter: bool) -> None:
        kw = {}
        if enable_combine_scatter:
            kw.update(combine_row_topk=row_to_topk,
                      combine_buffer_ptrs=combine_buffer_ptrs_t,
                      combine_tokens_per_rank=tokens_per_rank,
                      combine_top_k=combine_top_k)
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_pool, sfa_global), b_fp8, d, psum_layout,
            recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
            disable_ue8m0_cast=True,
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m_per_expert,
            gather_index=gather_index,
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

    def run_standalone_scatter() -> None:
        with torch.cuda.stream(compute_stream):
            deep_gemm.combine_scatter_copy_rows(
                d, gather_index, row_to_topk, combine_buffer_ptrs_t,
                tokens_per_rank, combine_top_k, args.scatter_rows_per_block)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_two_stage_scatter() -> None:
        with torch.cuda.stream(compute_stream):
            launch_gemm(enable_combine_scatter=False)
            deep_gemm.combine_scatter_copy_rows(
                d, gather_index, row_to_topk, combine_buffer_ptrs_t,
                tokens_per_rank, combine_top_k, args.scatter_rows_per_block)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_pack_for_reduce_scatter() -> None:
        with torch.cuda.stream(compute_stream):
            reduce_scatter_input.zero_()
            deep_gemm.combine_pack_for_reduce_scatter(
                d, gather_index, row_to_topk, combine_topk_scores, reduce_scatter_input,
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
                d, gather_index, row_to_topk, combine_topk_scores, reduce_scatter_input,
                tokens_per_rank, combine_top_k)
        with torch.cuda.stream(comm_stream):
            comm_stream.wait_stream(compute_stream)
            deep_gemm.nccl_reduce_scatter_sum(reduce_scatter_input, reduce_scatter_output, nccl_cpp_comm)
        torch.cuda.current_stream().wait_stream(compute_stream)
        torch.cuda.current_stream().wait_stream(comm_stream)

    def run_local_reduce_only() -> None:
        with torch.cuda.stream(compute_stream):
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
            combine_buffer.fill_(float('nan'))
            torch.cuda.synchronize()
            dist.barrier(group=group)
            fn()
            torch.cuda.synchronize()
            dist.barrier(group=group)
            max_diff_t, mismatch_count_t = deep_gemm.check_combine_scatter_output(
                d, gather_index, row_to_topk, combine_buffer_ptrs_t,
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

        check_scatter_output('Fused combine-scatter epilogue', run_fused_scatter)
        if args.bench_standalone_scatter:
            check_scatter_output('Standalone scatter-copy', run_standalone_scatter)

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
        if args.bench_reduce_scatter:
            print(f'  nccl_ctas={os.environ["NCCL_MIN_CTAS"]}/{os.environ["NCCL_MAX_CTAS"]}', flush=True)
        print('  common components:', flush=True)
        print(f'    GEMM no scatter       : {gemm_no_scatter_ms * 1e3:8.2f} us '
              f'(event {gemm_no_scatter_event_ms * 1e3:8.2f} us)', flush=True)
        print('  fused scatter scheme:', flush=True)
        print(f'    fused GEMM+scatter    : {fused_scatter_ms * 1e3:8.2f} us '
              f'(event {fused_scatter_event_ms * 1e3:8.2f} us)', flush=True)
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
    parser.add_argument('--bench-standalone-scatter', action='store_true',
                        help='Benchmark separate D -> peer combine-buffer scatter-copy and two-stage total')
    parser.add_argument('--bench-reduce-scatter', action='store_true',
                        help='Benchmark GEMM + local pack/reduce + NCCL reduce-scatter baseline')
    parser.add_argument('--scatter-rows-per-block', type=int, default=4,
                        choices=(1, 2, 4, 8, 16, 32),
                        help='Rows handled by each standalone scatter-copy CTA')
    parser.add_argument('--reduce-check-rtol', type=float, default=1e-2,
                        help='Relative tolerance for fused scatter+local-reduce vs reduce-scatter baseline')
    parser.add_argument('--reduce-check-atol', type=float, default=2e-2,
                        help='Absolute tolerance for fused scatter+local-reduce vs reduce-scatter baseline')
    parser.add_argument('--num-weights', type=int, default=1,
                        help='Number of weight matrices per expert (e.g. 2 for fused gate+up)')
    parser.add_argument('--block-m', type=int, default=128)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()

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
