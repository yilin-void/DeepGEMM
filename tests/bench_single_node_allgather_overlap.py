"""Bench single-node symmetric-memory all-gather overlapped with grouped GEMM.

Run example:
  python tests/bench_single_node_allgather_overlap.py --num-local-ranks 8

The timed all-gather payload is the FP8 activation tensor. SFA is prepared before
timing because the current Python/C++ GEMM API transforms SFA before launching
the rank-flag-aware GEMM kernel, so the GEMM-side flag wait cannot guard SFA
communication yet.
"""

import argparse
import contextlib
import os
import socket
import statistics
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem

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


def _generate_routing_topk(num_tokens: int, top_k: int, num_experts: int, seed: int) -> torch.Tensor:
    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    return torch.randint(0, num_experts, (num_tokens, top_k),
                         dtype=torch.int32, device='cuda', generator=gen)


def _generate_distinct_routing_topk(num_tokens: int, top_k: int, num_experts: int, seed: int) -> torch.Tensor:
    if top_k > num_experts:
        raise ValueError(f'top_k ({top_k}) must be <= num_experts ({num_experts})')

    gen = torch.Generator(device='cuda')
    gen.manual_seed(seed)
    scores = torch.rand((num_tokens, num_experts), dtype=torch.float32,
                        device='cuda', generator=gen)
    return torch.topk(scores, top_k, dim=1).indices.to(torch.int32)


def _should_profile_rank(arg: str, rank: int, num_ranks: int, cross_rank_sync: str) -> bool:
    if arg == 'all':
        return True
    if arg == 'auto':
        if cross_rank_sync == 'ipc-stream' and num_ranks >= 8:
            return rank == 0
        return True
    return rank in {int(x) for x in arg.split(',')}


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


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    rank, num_ranks, group = init_dist(local_rank, num_local_ranks)
    if get_arch_major() != 9:
        dist_print('SM90 is required for the current rank-flag grouped GEMM path.', once_in_node=True)
        return

    nccl_cpp_comm = None
    if args.allgather_backend == 'nccl-cpp':
        nccl_unique_ids = [deep_gemm.nccl_get_unique_id() if rank == 0 else None]
        dist.broadcast_object_list(nccl_unique_ids, src=0, group=group)
        nccl_cpp_comm = deep_gemm.nccl_comm_init_rank(nccl_unique_ids[0], rank, num_ranks, local_rank)
        dist.barrier(group=group)

    torch.manual_seed(0x1234 + rank)
    tokens_per_rank = args.tokens_per_rank
    total_tokens = num_ranks * tokens_per_rank
    hidden = args.hidden
    num_experts = args.num_experts
    top_k = args.top_k
    block_m = args.block_m
    routing_desc = 'random'

    if args.routing_mode == 'all-ranks-local':
        if args.global_num_experts is None:
            raise ValueError('--global-num-experts is required for --routing-mode all-ranks-local')
        if args.global_num_experts % num_ranks != 0:
            raise ValueError('--global-num-experts must be divisible by the EP/world size')
        if top_k != args.experts_per_rank_token * num_ranks:
            raise ValueError('--top-k must equal --experts-per-rank-token * EP/world size')

        num_experts = args.global_num_experts // num_ranks
        top_k = args.experts_per_rank_token
        routing_desc = (f'all-ranks-local: global_experts={args.global_num_experts}, '
                        f'global_top_k={args.top_k}, local_experts={num_experts}, '
                        f'local_top_k={top_k}')

    _rank0_print(rank, 'Preparing benchmark tensors...')
    quant_config = QuantConfig()
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    # Deterministic global activations let every rank precompute SFA outside the
    # timed region while the FP8 A payload itself is still gathered over NVLink.
    torch.manual_seed(0x2026)
    a_global_bf16 = torch.randn((total_tokens, hidden), dtype=torch.bfloat16, device='cuda')
    a_global_fp8, sfa_global = cast_fp8_fp4_with_major(
        a_global_bf16, MajorTypeAB.KMajor, quant_config.gran_k_a,
        quant_config.is_fp4_a, use_ue8m0=False)
    a_local = a_global_fp8[rank * tokens_per_rank:(rank + 1) * tokens_per_rank].contiguous()

    # Output buffer for the all-gathered A pool. The original overlap path uses
    # symmetric memory so peers can P2P-pull rank slots and publish per-rank
    # ready flags. The NCCL C++ backend is for serial baseline measurement only:
    # the collective completes before GEMM, so no rank_flags are needed by GEMM.
    if args.allgather_backend == 'symm':
        a_symm = symm_mem.empty((num_ranks, tokens_per_rank, hidden),
                                dtype=a_local.dtype, device='cuda')
        a_handle = symm_mem.rendezvous(a_symm, group=group)
        a_pool = a_symm.view(total_tokens, hidden)
        a_buffer_ptrs = [int(p) for p in a_handle.buffer_ptrs]
    else:
        a_symm = None
        a_handle = None
        a_pool = torch.empty((total_tokens, hidden), dtype=a_local.dtype, device='cuda')
        a_buffer_ptrs = None

    # Grouped GEMM metadata and weights.
    _rank0_print(rank, 'Building gather layout...')
    if args.routing_mode == 'all-ranks-local':
        routing_topk = _generate_distinct_routing_topk(total_tokens, top_k, num_experts, seed=0xBEEF)
        topk_slot_offset = rank * top_k
    else:
        routing_topk = _generate_routing_topk(total_tokens, top_k, num_experts, seed=0xBEEF)
        topk_slot_offset = 0
    gather_index, tile_rank, grouped_layout, m_logical_t, psum_layout, _row_to_topk = \
        deep_gemm.build_gather_layout_for_rank_overlap(
            routing_topk, rank, num_ranks, tokens_per_rank, num_experts, block_m,
            topk_slot_offset=topk_slot_offset)
    m_logical = int(m_logical_t.item())
    num_m_tiles = (m_logical + block_m - 1) // block_m
    expected_m_per_expert = int((m_logical + num_experts - 1) // num_experts * 1.2)
    _rank0_print(rank, f'Gather layout ready: m_logical={m_logical}, tiles={num_m_tiles}')

    _rank0_print(rank, 'Preparing grouped GEMM weights...')
    n_eff = args.n * args.num_weights
    torch.manual_seed(0x5678 + rank)
    b_bf16 = torch.randn((num_experts, n_eff, hidden), dtype=torch.bfloat16, device='cuda')
    b_fp8 = grouped_cast_fp8_fp4_with_major(
        b_bf16, MajorTypeAB.KMajor, quant_config.gran_k_b,
        quant_config.is_fp4_b, use_ue8m0=False, use_block_cast_for_fp8=True)

    d = torch.empty((m_logical, n_eff), dtype=torch.bfloat16, device='cuda')
    rank_flags = torch.empty((num_ranks,), dtype=torch.int64, device='cuda')
    ready_flag = None
    ready_write_base_ptrs = None
    ready_wait_ptrs = None
    if args.allgather_backend == 'symm' and args.cross_rank_sync == 'ipc-stream':
        # Each rank owns a local ready vector. After rank r finishes its local
        # D2D copy, it writes ready[r] on every peer via P2P stream mem-op.
        # Each peer waits only on its own local ready[src_rank] slot before
        # pulling that src rank.
        ready_flag = deep_gemm.cuda_ipc_alloc_i64(num_ranks)
        ready_handles = [None] * num_ranks
        dist.all_gather_object(ready_handles, deep_gemm.cuda_ipc_get_mem_handle(ready_flag), group=group)
        ready_write_base_ptrs = deep_gemm.cuda_ipc_open_mem_handles(ready_handles, rank, ready_flag)
        ready_wait_ptrs = [
            ready_write_base_ptrs[rank] + src_rank * ready_flag.element_size()
            for src_rank in range(num_ranks)
        ]

    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()
    flags_zeroed = torch.cuda.Event()
    local_copy_done = torch.cuda.Event()
    comm_ready = torch.cuda.Event()
    ready_epoch = 0

    def get_overlap_num_sms(base_num_sms: int) -> int:
        if args.overlap_reserved_sms <= 0:
            return base_num_sms
        overlap_num_sms = max(2, base_num_sms - args.overlap_reserved_sms)
        if overlap_num_sms % 2 != 0:
            overlap_num_sms -= 1
        return overlap_num_sms

    def get_overlap_reserved_sms() -> int:
        if args.overlap_launch_order == 'gemm-first':
            return args.overlap_reserved_sms
        if args.overlap_launch_order == 'after-local-copy' and args.cross_rank_sync == 'barrier':
            return args.overlap_reserved_sms
        return 0

    def get_reported_overlap_num_sms(base_num_sms: int) -> int:
        return get_overlap_num_sms(base_num_sms) if get_overlap_reserved_sms() > 0 else base_num_sms

    def launch_gemm(with_rank_flags=True, reserved_sms: int = 0, epoch: int = 0) -> None:
        kw = {}
        if with_rank_flags:
            kw.update(rank_flags=rank_flags, tile_rank=tile_rank, num_ranks=num_ranks,
                      rank_flag_epoch=epoch)
        old_num_sms = deep_gemm.get_num_sms()
        if reserved_sms > 0:
            deep_gemm.set_num_sms(get_overlap_num_sms(old_num_sms))
        try:
            deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
                (a_pool, sfa_global), b_fp8, d, psum_layout,
                recipe=recipe, recipe_a=recipe_a, recipe_b=recipe_b,
                disable_ue8m0_cast=True,
                use_psum_layout=True,
                expected_m_for_psum_layout=expected_m_per_expert,
                gather_index=gather_index,
                **kw,
            )
        finally:
            if reserved_sms > 0:
                deep_gemm.set_num_sms(old_num_sms)

    def run_nccl_allgather() -> None:
        with torch.cuda.stream(comm_stream):
            deep_gemm.nccl_allgather_bytes(a_local, a_pool, nccl_cpp_comm)

    def run_symm_allgather(epoch: int) -> None:
        deep_gemm.single_node_allgather_copy_local(
            [a_local], [a_symm], rank, rank_flags, flag_value=epoch)
        publish_local_ready(epoch)
        pull_after_ready(epoch)

    def pull_after_barrier() -> None:
        a_handle.barrier()
        deep_gemm.single_node_allgather_pull([a_symm], [a_buffer_ptrs], rank, num_ranks, rank_flags)

    def next_ready_epoch() -> int:
        nonlocal ready_epoch
        ready_epoch += 1
        return ready_epoch

    def publish_local_ready(epoch: int) -> None:
        if args.cross_rank_sync == 'ipc-stream':
            for dst_rank in range(num_ranks):
                ready_ptr = ready_write_base_ptrs[dst_rank] + rank * ready_flag.element_size()
                deep_gemm.stream_write_value64_ptr(ready_ptr, epoch)

    def pull_after_ready(epoch: int) -> None:
        if args.cross_rank_sync == 'ipc-stream':
            deep_gemm.single_node_allgather_pull_with_ready_flags(
                [a_symm], [a_buffer_ptrs], ready_wait_ptrs,
                rank, num_ranks, rank_flags, flag_value=epoch,
                ready_value=epoch)
        else:
            pull_after_barrier()

    def run_allgather_only() -> None:
        if args.allgather_backend == 'nccl-cpp':
            run_nccl_allgather()
            torch.cuda.current_stream().wait_stream(comm_stream)
        else:
            epoch = next_ready_epoch()
            with torch.cuda.stream(comm_stream):
                run_symm_allgather(epoch)
            torch.cuda.current_stream().wait_stream(comm_stream)

    def run_gemm_only() -> None:
        if args.allgather_backend == 'nccl-cpp':
            run_nccl_allgather()
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_stream(comm_stream)
                launch_gemm(with_rank_flags=False)
            torch.cuda.current_stream().wait_stream(comm_stream)
            torch.cuda.current_stream().wait_stream(compute_stream)
        else:
            epoch = next_ready_epoch()
            with torch.cuda.stream(comm_stream):
                run_symm_allgather(epoch)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_stream(comm_stream)
                launch_gemm(epoch=epoch)
            torch.cuda.current_stream().wait_stream(comm_stream)
            torch.cuda.current_stream().wait_stream(compute_stream)

    def run_gemm_no_flags() -> None:
        with torch.cuda.stream(compute_stream):
            launch_gemm(with_rank_flags=False)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_serial() -> None:
        if args.allgather_backend == 'nccl-cpp':
            run_nccl_allgather()
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_stream(comm_stream)
                launch_gemm(with_rank_flags=False)
            torch.cuda.current_stream().wait_stream(comm_stream)
            torch.cuda.current_stream().wait_stream(compute_stream)
        else:
            epoch = next_ready_epoch()
            with torch.cuda.stream(comm_stream):
                run_symm_allgather(epoch)
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_stream(comm_stream)
                launch_gemm(epoch=epoch)
            torch.cuda.current_stream().wait_stream(comm_stream)
            torch.cuda.current_stream().wait_stream(compute_stream)

    def run_overlap_allgather_first() -> None:
        epoch = next_ready_epoch()
        if args.cross_rank_sync == 'barrier':
            with torch.cuda.stream(comm_stream):
                rank_flags.zero_()
                deep_gemm.single_node_allgather_copy_local([a_local], [a_symm], rank, rank_flags)
                a_handle.barrier()
                comm_ready.record()
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(comm_ready)
                launch_gemm(epoch=epoch)
            with torch.cuda.stream(comm_stream):
                pull_after_barrier()
        else:
            with torch.cuda.stream(comm_stream):
                deep_gemm.single_node_allgather_copy_local(
                    [a_local], [a_symm], rank, rank_flags, flag_value=epoch)
            with torch.cuda.stream(compute_stream):
                launch_gemm(epoch=epoch)
            with torch.cuda.stream(comm_stream):
                publish_local_ready(epoch)
                pull_after_ready(epoch)
        torch.cuda.current_stream().wait_stream(comm_stream)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_overlap_gemm_first() -> None:
        epoch = next_ready_epoch()
        if args.cross_rank_sync == 'barrier':
            with torch.cuda.stream(comm_stream):
                rank_flags.zero_()
                flags_zeroed.record()
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(flags_zeroed)
                launch_gemm(reserved_sms=args.overlap_reserved_sms, epoch=epoch)
            with torch.cuda.stream(comm_stream):
                deep_gemm.single_node_allgather_copy_local([a_local], [a_symm], rank, rank_flags)
                a_handle.barrier()
                deep_gemm.single_node_allgather_pull(
                    [a_symm], [a_buffer_ptrs], rank, num_ranks, rank_flags)
        else:
            with torch.cuda.stream(compute_stream):
                launch_gemm(reserved_sms=args.overlap_reserved_sms, epoch=epoch)
            with torch.cuda.stream(comm_stream):
                deep_gemm.single_node_allgather_copy_local(
                    [a_local], [a_symm], rank, rank_flags, flag_value=epoch)
                publish_local_ready(epoch)
                pull_after_ready(epoch)
        torch.cuda.current_stream().wait_stream(comm_stream)
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_overlap_after_local_copy() -> None:
        epoch = next_ready_epoch()
        if args.cross_rank_sync == 'barrier':
            with torch.cuda.stream(comm_stream):
                rank_flags.zero_()
                deep_gemm.single_node_allgather_copy_local([a_local], [a_symm], rank, rank_flags)
                publish_local_ready(epoch)
                local_copy_done.record()
            with torch.cuda.stream(compute_stream):
                compute_stream.wait_event(local_copy_done)
                launch_gemm(reserved_sms=get_overlap_reserved_sms(), epoch=epoch)
            with torch.cuda.stream(comm_stream):
                pull_after_ready(epoch)
        else:
            with torch.cuda.stream(comm_stream):
                deep_gemm.single_node_allgather_copy_local(
                    [a_local], [a_symm], rank, rank_flags, flag_value=epoch)
                publish_local_ready(epoch)
                pull_after_ready(epoch)
            with torch.cuda.stream(compute_stream):
                launch_gemm(reserved_sms=get_overlap_reserved_sms(), epoch=epoch)
        torch.cuda.current_stream().wait_stream(comm_stream)
        torch.cuda.current_stream().wait_stream(compute_stream)

    overlap_fns = {
        'gemm-first': run_overlap_gemm_first,
        'after-local-copy': run_overlap_after_local_copy,
        'allgather-first': run_overlap_allgather_first,
    }
    run_overlap = run_serial if args.allgather_backend == 'nccl-cpp' else overlap_fns[args.overlap_launch_order]

    # Compile/warm the JIT paths, fill A once, and optionally validate all-gather.
    _rank0_print(rank, 'Warming all-gather path...')
    run_allgather_only()
    torch.cuda.synchronize()
    if args.check:
        _rank0_print(rank, 'Checking all-gather result...')
        torch.testing.assert_close(a_pool, a_global_fp8, rtol=0, atol=0)

    if args.profile:
        profile_iters = args.profile_iters
        profile_dir = args.profile_dir
        os.makedirs(profile_dir, exist_ok=True)

        do_profile = _should_profile_rank(args.profile_ranks, rank, num_ranks, args.cross_rank_sync)
        if do_profile:
            dist_print(f'Rank {rank}: profiler enabled', once_in_node=False)
        else:
            dist_print(f'Rank {rank}: profiler disabled (--profile-ranks={args.profile_ranks})',
                       once_in_node=False)

        _rank0_print(rank, f'Profiling config: allgather_backend={args.allgather_backend}')
        if args.allgather_backend == 'nccl-cpp':
            _rank0_print(rank, f'Profiling config: nccl_ctas={os.environ["NCCL_MIN_CTAS"]}/{os.environ["NCCL_MAX_CTAS"]}')
        _rank0_print(rank, 'Profiling timeline (warmup + capture)...')

        # Warmup outside profiler to avoid JIT noise.
        # dist.barrier between iterations prevents epoch-overwrite race:
        # a fast rank's next-epoch cuStreamWriteValue64 can overwrite the
        # ready_flag before a slow rank's cuStreamWaitValue64 polls it.
        for fn in (run_allgather_only, run_gemm_only, run_serial, run_overlap):
            for _ in range(2):
                fn()
                torch.cuda.synchronize()
                dist.barrier(group=group)

        prof_ctx = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
        ) if do_profile else contextlib.nullcontext()

        with prof_ctx as prof:
            for label, fn in [('allgather_only', run_allgather_only),
                              ('gemm_only', run_gemm_only),
                              ('serial', run_serial),
                              ('overlap', run_overlap)]:
                for i in range(profile_iters):
                    dist.barrier(group=group)
                    torch.cuda.synchronize()
                    rec_ctx = (torch.profiler.record_function(f'{label}/iter_{i}')
                               if do_profile else contextlib.nullcontext())
                    with rec_ctx:
                        fn()
                    torch.cuda.synchronize()
                dist.barrier(group=group)

        if do_profile:
            trace_path = os.path.join(profile_dir, f'ag_overlap_rank{rank}.json')
            prof.export_chrome_trace(trace_path)
        _rank0_print(rank, f'Trace written to {profile_dir}/')
    else:
        _rank0_print(rank, 'Benchmarking all-gather only...')
        comm_ms = _max_across_ranks(_bench_ms(run_allgather_only, group, warmups=args.warmups, iters=args.iters), group)
        comm_event_ms = None
        if args.allgather_backend == 'nccl-cpp':
            comm_event_ms = _max_across_ranks(
                _bench_cuda_event_ms(run_nccl_allgather, group, comm_stream,
                                     warmups=args.warmups, iters=args.iters),
                group)
        _rank0_print(rank, 'Benchmarking GEMM (no rank_flags)...')
        gemm_nf_ms = _max_across_ranks(_bench_ms(run_gemm_no_flags, group, warmups=args.warmups, iters=args.iters), group)
        gemm_nf_event_ms = _max_across_ranks(
            _bench_cuda_event_ms(run_gemm_no_flags, group, compute_stream,
                                 warmups=args.warmups, iters=args.iters),
            group)
        gemm_ms = None
        if args.allgather_backend == 'symm':
            _rank0_print(rank, 'Benchmarking GEMM (with rank_flags)...')
            gemm_ms = _max_across_ranks(_bench_ms(run_gemm_only, group, warmups=args.warmups, iters=args.iters), group)
        _rank0_print(rank, 'Benchmarking serial all-gather + GEMM...')
        serial_ms = _max_across_ranks(_bench_ms(run_serial, group, warmups=args.warmups, iters=args.iters), group)
        overlap_ms = None
        if args.allgather_backend == 'symm':
            _rank0_print(rank, 'Benchmarking overlapped all-gather + GEMM...')
            overlap_ms = _max_across_ranks(_bench_ms(run_overlap, group, warmups=args.warmups, iters=args.iters), group)

        if rank == 0:
            payload_mb = a_local.numel() * a_local.element_size() / 1e6
            print('Single-node all-gather + grouped GEMM bench:', flush=True)
            print(f'  ranks={num_ranks}, tokens/rank={tokens_per_rank}, hidden={hidden}, '
                  f'payload/rank={payload_mb:.1f} MB', flush=True)
            print(f'  routing={routing_desc}', flush=True)
            print(f'  allgather_backend={args.allgather_backend}', flush=True)
            if args.allgather_backend == 'nccl-cpp':
                print(f'  nccl_ctas={os.environ["NCCL_MIN_CTAS"]}/{os.environ["NCCL_MAX_CTAS"]}', flush=True)
            print(f'  overlap_launch_order={args.overlap_launch_order}, '
                  f'cross_rank_sync={args.cross_rank_sync}, '
                  f'overlap_reserved_sms={args.overlap_reserved_sms}, '
                  f'overlap_gemm_sms={get_reported_overlap_num_sms(deep_gemm.get_num_sms())}', flush=True)
            print(f'  experts={num_experts}, top_k={top_k}, m_logical={m_logical}, '
                  f'n={args.n}, num_weights={args.num_weights}, n_eff={n_eff}', flush=True)
            print('  common components:', flush=True)
            print(f'    allgather only        : {comm_ms * 1e3:8.2f} us', flush=True)
            if comm_event_ms is not None:
                print(f'    allgather event       : {comm_event_ms * 1e3:8.2f} us', flush=True)
            print(f'    GEMM no rank_flags    : {gemm_nf_ms * 1e3:8.2f} us '
                  f'(event {gemm_nf_event_ms * 1e3:8.2f} us)', flush=True)
            if gemm_ms is None:
                print('  GEMM (w/ flags):      n/a (NCCL C++ backend has no rank_flags)', flush=True)
            else:
                print(f'  GEMM (w/ flags): {gemm_ms * 1e3:8.2f} us', flush=True)
            print(f'  serial         : {serial_ms * 1e3:8.2f} us', flush=True)
            if overlap_ms is None:
                print('  overlap        :      n/a (NCCL C++ backend is serial collective)', flush=True)
            else:
                print(f'  overlap        : {overlap_ms * 1e3:8.2f} us', flush=True)
            if serial_ms > 0 and overlap_ms is not None:
                print(f'  speedup        : {serial_ms / overlap_ms:8.3f}x', flush=True)

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-local-ranks', type=int, default=None)
    parser.add_argument('--tokens-per-rank', type=int, default=1024)
    parser.add_argument('--hidden', type=int, default=7168)
    parser.add_argument('--n', type=int, default=4096)
    parser.add_argument('--num-experts', type=int, default=8)
    parser.add_argument('--top-k', type=int, default=2)
    parser.add_argument('--routing-mode', type=str, default='random',
                        choices=('random', 'all-ranks-local'),
                        help='random: top-k local experts; all-ranks-local: global top-k split evenly across EP ranks')
    parser.add_argument('--global-num-experts', type=int, default=None,
                        help='Global expert count for all-ranks-local routing')
    parser.add_argument('--experts-per-rank-token', type=int, default=2,
                        help='Number of local experts selected per token on each EP rank')
    parser.add_argument('--overlap-launch-order', type=str, default='allgather-first',
                        choices=('gemm-first', 'after-local-copy', 'allgather-first'),
                        help='Whether overlap mode submits GEMM before or after all-gather pull work')
    parser.add_argument('--cross-rank-sync', type=str, default='barrier',
                        choices=('barrier', 'ipc-stream'),
                        help='barrier: use symmetric-memory global barrier; ipc-stream: per-rank CUDA IPC ready flags with cuStreamWait/WriteValue64')
    parser.add_argument('--allgather-backend', type=str, default='symm',
                        choices=('symm', 'nccl-cpp'),
                        help='symm: custom symmetric-memory P2P all-gather; '
                             'nccl-cpp: use a direct C++ ncclAllGather wrapper')
    parser.add_argument('--overlap-reserved-sms', type=int, default=2,
                        help='SMs left unused by GEMM when the selected overlap mode needs GPU-side communication progress')
    parser.add_argument('--num-weights', type=int, default=1,
                        help='Number of weight matrices per expert (e.g. 2 for fused gate+up)')
    parser.add_argument('--block-m', type=int, default=128)
    parser.add_argument('--warmups', type=int, default=3)
    parser.add_argument('--iters', type=int, default=10)
    parser.add_argument('--check', action='store_true')
    parser.add_argument('--profile', action='store_true',
                        help='Capture torch profiler timeline and export Chrome trace JSON')
    parser.add_argument('--profile-iters', type=int, default=3,
                        help='Number of iterations per mode to capture in the profile')
    parser.add_argument('--profile-dir', type=str, default='profile_traces',
                        help='Directory to write trace JSON files')
    parser.add_argument('--profile-ranks', type=str, default='auto',
                        help='Comma-separated rank IDs to enable torch.profiler on, '
                             'or "all"/"auto". auto: profile all ranks unless '
                             'ipc-stream + >=8 ranks (then rank 0 only)')
    args = parser.parse_args()

    if args.allgather_backend == 'nccl-cpp':
        _set_default_nccl_ctas()

    num_local_ranks = args.num_local_ranks or torch.cuda.device_count()
    assert num_local_ranks > 0
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = str(_find_free_port())
        print(f'Using MASTER_PORT={os.environ["MASTER_PORT"]}', flush=True)
    torch.multiprocessing.spawn(_worker, args=(num_local_ranks, args), nprocs=num_local_ranks)


if __name__ == '__main__':
    main()
