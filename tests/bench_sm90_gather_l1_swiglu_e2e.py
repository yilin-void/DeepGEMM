"""End-to-end SM90 MoE benchmark for gather-L1 + SwiGLU fusion.

Both paths use the same direct C++ NCCL all-gather dispatch, selected BF16 or
FP8 L2 combine-scatter epilogue, and matching local scored reduction. They
differ only in whether gather-L1 and SwiGLU quantization are separate kernels
or fused.
"""

import argparse
import os
import socket
import statistics
import sys
import time
from typing import Callable

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import deep_gemm

# generators.py is shared by installations whose package name is either
# deep_gemm or deep_gemm_moe_L1.
sys.modules.setdefault("deep_gemm_moe_L1", deep_gemm)
sys.path.insert(0, os.path.dirname(__file__))
from generators import (  # noqa: E402
    MajorTypeAB,
    QuantConfig,
    cast_fp8_fp4_with_major,
    grouped_cast_fp8_fp4_with_major,
)
from sm90_gather_l1_swiglu_baseline import (  # noqa: E402
    allocate_swiglu_output,
    swiglu_quant_fp8,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _set_default_nccl_ctas() -> None:
    os.environ.setdefault("NCCL_MIN_CTAS", "64")
    os.environ.setdefault("NCCL_MAX_CTAS", "64")


def _make_routing(total_tokens: int, num_ranks: int, local_experts: int,
                  experts_per_rank_token: int) -> torch.Tensor:
    generator = torch.Generator(device="cuda").manual_seed(0xBEEF)
    scores = torch.rand(
        (total_tokens, num_ranks, local_experts),
        device="cuda",
        dtype=torch.float32,
        generator=generator,
    )
    choices = torch.topk(scores, experts_per_rank_token, dim=2).indices
    offsets = torch.arange(num_ranks, device="cuda", dtype=torch.int64)
    choices = choices + offsets.view(1, num_ranks, 1) * local_experts
    return choices.reshape(total_tokens, -1).to(torch.int32)


def _make_wgmma_n64_physical_to_logical_index(
    n: int, device: torch.device
) -> torch.Tensor:
    if n % 64 != 0:
        raise ValueError(f"N must be divisible by 64 for FP8 combine-scatter, got {n}")
    physical = torch.arange(64, device=device)
    half = physical // 32
    in_half = physical % 32
    pair = in_half // 8
    lane_group = (in_half % 8) // 2
    elem = in_half % 2
    logical = lane_group * 16 + half * 8 + pair * 2 + elem
    tile_base = torch.arange(0, n, 64, device=device).unsqueeze(1)
    return (tile_base + logical.unsqueeze(0)).reshape(-1).to(torch.long)


def _make_wgmma_n32_physical_to_logical_index(
    n: int, device: torch.device
) -> torch.Tensor:
    if n % 32 != 0:
        raise ValueError(f"N must be divisible by 32 for BF16 combine-scatter, got {n}")
    physical = torch.arange(32, device=device)
    pair = physical // 8
    lane_group = (physical % 8) // 2
    elem = physical % 2
    logical = lane_group * 8 + pair * 2 + elem
    tile_base = torch.arange(0, n, 32, device=device).unsqueeze(1)
    return (tile_base + logical.unsqueeze(0)).reshape(-1).to(torch.long)


def _bench(
    fn: Callable[[], None],
    warmups: int,
    iters: int,
    group: dist.ProcessGroup,
) -> list[float]:
    for _ in range(warmups):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        dist.barrier(group=group)

    times = []
    for _ in range(iters):
        dist.barrier(group=group)
        torch.cuda.synchronize()
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - start) * 1e3
        dist.barrier(group=group)
        times.append(elapsed_ms)
    return times


def _bench_balanced_pair(
    baseline_fn: Callable[[], None],
    fused_fn: Callable[[], None],
    warmups: int,
    iters: int,
    group: dist.ProcessGroup,
) -> tuple[list[float], list[float]]:
    baseline_first = _bench(baseline_fn, warmups, iters, group)
    fused_second = _bench(fused_fn, warmups, iters, group)
    fused_first = _bench(fused_fn, warmups, iters, group)
    baseline_second = _bench(baseline_fn, warmups, iters, group)
    return baseline_first + baseline_second, fused_first + fused_second


def _check_final_output(
    actual: torch.Tensor,
    expected: torch.Tensor,
    group: dist.ProcessGroup,
) -> tuple[float, float, float, float, float]:
    sums = torch.zeros(6, device=actual.device, dtype=torch.float64)
    max_abs = torch.zeros(1, device=actual.device, dtype=torch.float64)
    nonfinite = torch.zeros(1, device=actual.device, dtype=torch.int64)
    for row_start in range(0, actual.shape[0], 512):
        row_end = min(row_start + 512, actual.shape[0])
        lhs = actual[row_start:row_end]
        rhs = expected[row_start:row_end]
        finite = torch.isfinite(lhs) & torch.isfinite(rhs)
        nonfinite += (~finite).sum()
        lhs64 = torch.where(finite, lhs, 0.0).double()
        rhs64 = torch.where(finite, rhs, 0.0).double()
        error = (lhs64 - rhs64).abs()
        sums[0] += (lhs64 * rhs64).sum()
        sums[1] += (lhs64.square() + rhs64.square()).sum()
        sums[2] += error.sum()
        sums[3] += (lhs != rhs).sum()
        sums[4] += lhs.numel()
        sums[5] += rhs64.abs().sum()
        max_abs = torch.maximum(max_abs, error.max().reshape(1))

    dist.all_reduce(sums, op=dist.ReduceOp.SUM, group=group)
    dist.all_reduce(max_abs, op=dist.ReduceOp.MAX, group=group)
    dist.all_reduce(nonfinite, op=dist.ReduceOp.SUM, group=group)
    if int(nonfinite.item()) != 0:
        raise AssertionError(f"final output contains {int(nonfinite.item())} non-finite pairs")

    normalized_diff = 0.0 if sums[1].item() == 0 else 1.0 - 2.0 * sums[0].item() / sums[1].item()
    mean_abs = sums[2].item() / sums[4].item()
    mismatch_rate = sums[3].item() / sums[4].item()
    reference_abs_mean = sums[5].item() / sums[4].item()
    return normalized_diff, mean_abs, max_abs.item(), mismatch_rate, reference_abs_mean


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{args.port}",
        rank=local_rank,
        world_size=num_local_ranks,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    group = dist.group.WORLD
    rank = dist.get_rank(group)
    num_ranks = dist.get_world_size(group)

    if args.global_num_experts % num_ranks != 0:
        raise ValueError("global-num-experts must be divisible by num-local-ranks")
    if args.top_k != args.experts_per_rank_token * num_ranks:
        raise ValueError("top-k must equal experts-per-rank-token * num-local-ranks")
    if args.hidden % 128 != 0 or args.intermediate % 128 != 0:
        raise ValueError("hidden and intermediate must be divisible by 128")
    if args.scatter_dtype not in ("bf16", "fp8"):
        raise ValueError("worker scatter-dtype must be bf16 or fp8")
    scatter_fp8 = args.scatter_dtype == "fp8"

    unique_ids = [deep_gemm.nccl_get_unique_id() if rank == 0 else None]
    dist.broadcast_object_list(unique_ids, src=0, group=group)
    nccl_comm = deep_gemm.nccl_comm_init_rank(
        unique_ids[0], rank, num_ranks, local_rank
    )
    dist.barrier(group=group)

    local_experts = args.global_num_experts // num_ranks
    total_tokens = args.tokens_per_rank * num_ranks
    quant = QuantConfig()

    torch.manual_seed(0x2026)
    a_global_bf16 = torch.randn(
        (total_tokens, args.hidden), device="cuda", dtype=torch.bfloat16
    )
    a_global, sfa_global = cast_fp8_fp4_with_major(
        a_global_bf16,
        MajorTypeAB.KMajor,
        quant.gran_k_a,
        quant.is_fp4_a,
        False,
    )
    del a_global_bf16
    token_start = rank * args.tokens_per_rank
    token_end = token_start + args.tokens_per_rank
    a_local = a_global[token_start:token_end].contiguous()
    sfa_local = sfa_global[token_start:token_end].contiguous()
    a_pool = torch.empty_like(a_global)
    sfa_pool = torch.empty_like(sfa_global) if args.dispatch_scales else sfa_global

    routing = _make_routing(
        total_tokens, num_ranks, local_experts, args.experts_per_rank_token
    )
    gather_index, tile_rank, _, shape_m_tensor, psum_layout, row_to_topk = (
        deep_gemm.build_gather_layout_for_rank_overlap(
            routing,
            rank,
            num_ranks,
            args.tokens_per_rank,
            local_experts,
            128,
            topk_slot_offset=0,
            expert_srank_padding=False,
        )
    )
    shape_m = int(shape_m_tensor.item())
    num_tiles = (shape_m + 127) // 128
    gather_index = gather_index[:shape_m]
    row_to_topk = row_to_topk[:shape_m]
    tile_rank = tile_rank[:num_tiles]
    del routing

    torch.manual_seed(0x1234 + rank)
    l1_weight_bf16 = torch.randn(
        (local_experts, 2 * args.intermediate, args.hidden),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    l1_weights = grouped_cast_fp8_fp4_with_major(
        l1_weight_bf16,
        MajorTypeAB.KMajor,
        quant.gran_k_b,
        quant.is_fp4_b,
        False,
        use_block_cast_for_fp8=True,
    )
    del l1_weight_bf16
    fused_l1_weights = deep_gemm.transform_l1_weights_for_swiglu(l1_weights)

    torch.manual_seed(0x5678 + rank)
    l2_weight_bf16 = torch.randn(
        (local_experts, args.hidden, args.intermediate),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    l2_permutation = (
        _make_wgmma_n64_physical_to_logical_index
        if scatter_fp8
        else _make_wgmma_n32_physical_to_logical_index
    )(args.hidden, l2_weight_bf16.device)
    l2_weight_bf16 = l2_weight_bf16.index_select(1, l2_permutation).contiguous()
    l2_weights = grouped_cast_fp8_fp4_with_major(
        l2_weight_bf16,
        MajorTypeAB.KMajor,
        quant.gran_k_b,
        quant.is_fp4_b,
        False,
        use_block_cast_for_fp8=True,
    )
    del l2_weight_bf16, l2_permutation

    l1_output = torch.empty(
        (shape_m, 2 * args.intermediate), device="cuda", dtype=torch.bfloat16
    )
    activation, activation_scale = allocate_swiglu_output(
        shape_m, args.intermediate, l1_output.device
    )
    l2_output_scratch = torch.empty(
        (shape_m, args.hidden), device="cuda", dtype=torch.bfloat16
    )

    torch.manual_seed(0x3456)
    topk_scores = torch.rand(
        (total_tokens, args.top_k), device="cuda", dtype=torch.float32
    )
    local_topk_scores = topk_scores[token_start:token_end].contiguous()
    del topk_scores

    combine_buffer = torch.empty(
        (args.tokens_per_rank, args.top_k, args.hidden),
        device="cuda",
        dtype=torch.float8_e4m3fn if scatter_fp8 else torch.bfloat16,
    )
    combine_scales = None
    if scatter_fp8:
        combine_scales = torch.empty(
            (args.tokens_per_rank, args.top_k, args.hidden // 32),
            device="cuda",
            dtype=torch.float32,
        )
    combine_handles = [None] * num_ranks
    dist.all_gather_object(
        combine_handles, deep_gemm.cuda_ipc_get_mem_handle(combine_buffer), group=group
    )
    combine_ptrs = deep_gemm.cuda_ipc_open_mem_handles(
        combine_handles, rank, combine_buffer
    )
    combine_ptrs_tensor = torch.tensor(combine_ptrs, device="cuda", dtype=torch.int64)
    scale_ptrs_tensor = None
    if scatter_fp8:
        scale_handles = [None] * num_ranks
        dist.all_gather_object(
            scale_handles, deep_gemm.cuda_ipc_get_mem_handle(combine_scales), group=group
        )
        scale_ptrs = deep_gemm.cuda_ipc_open_mem_handles(
            scale_handles, rank, combine_scales
        )
        scale_ptrs_tensor = torch.tensor(scale_ptrs, device="cuda", dtype=torch.int64)
    final_output = torch.empty(
        (args.tokens_per_rank, args.hidden), device="cuda", dtype=torch.float32
    )

    expected_m = int((shape_m + local_experts - 1) // local_experts * 1.2)
    rank_flags = torch.ones((num_ranks,), device="cuda", dtype=torch.int64)
    l1_kwargs = dict(
        recipe=(1, 128, 128),
        disable_ue8m0_cast=True,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
        gather_index=gather_index,
        rank_flags=rank_flags,
        tile_rank=tile_rank,
        num_ranks=num_ranks,
        rank_flag_epoch=1,
    )
    l2_kwargs = dict(
        recipe=(1, 128, 128),
        disable_ue8m0_cast=True,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
        combine_src_index=gather_index,
        combine_row_topk=row_to_topk,
        combine_buffer_ptrs=combine_ptrs_tensor,
        combine_tokens_per_rank=args.tokens_per_rank,
        combine_top_k=args.top_k,
        combine_scatter_direct_accum_stg=True,
        combine_scatter_fp8=scatter_fp8,
        use_tma_store=False,
    )
    if scatter_fp8:
        l2_kwargs["combine_scale_ptrs"] = scale_ptrs_tensor

    comm_stream = torch.cuda.Stream()
    compute_stream = torch.cuda.Stream()

    def enqueue_dispatch() -> None:
        with torch.cuda.stream(comm_stream):
            deep_gemm.nccl_allgather_bytes(a_local, a_pool, nccl_comm)
            if args.dispatch_scales:
                deep_gemm.nccl_allgather_bytes(sfa_local, sfa_pool, nccl_comm)

    def launch_baseline_middle() -> None:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_pool, sfa_pool), l1_weights, l1_output, psum_layout, **l1_kwargs
        )
        swiglu_quant_fp8(l1_output, activation, activation_scale)

    def launch_fused_middle() -> None:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_pool, sfa_pool),
            fused_l1_weights,
            activation,
            psum_layout,
            swiglu_output_scale=activation_scale,
            **l1_kwargs,
        )

    def launch_l2_scatter() -> None:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (activation, activation_scale),
            l2_weights,
            l2_output_scratch,
            psum_layout,
            **l2_kwargs,
        )

    def launch_local_combine() -> None:
        if scatter_fp8:
            deep_gemm.combine_reduce_slots_fp8(
                combine_buffer, combine_scales, local_topk_scores, final_output
            )
        else:
            deep_gemm.combine_reduce_slots(
                combine_buffer, local_topk_scores, final_output
            )

    def run_dispatch() -> None:
        enqueue_dispatch()
        torch.cuda.current_stream().wait_stream(comm_stream)

    def run_baseline_middle() -> None:
        with torch.cuda.stream(compute_stream):
            launch_baseline_middle()
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_fused_middle() -> None:
        with torch.cuda.stream(compute_stream):
            launch_fused_middle()
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_l2_scatter() -> None:
        with torch.cuda.stream(compute_stream):
            launch_l2_scatter()
        # Every rank must finish its outbound P2P stores before any rank reads
        # its local combine slots.
        compute_stream.synchronize()
        dist.barrier(group=group)

    def run_local_combine() -> None:
        with torch.cuda.stream(compute_stream):
            launch_local_combine()
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_compute_chain(middle: Callable[[], None]) -> None:
        with torch.cuda.stream(compute_stream):
            middle()
            launch_l2_scatter()
        compute_stream.synchronize()
        dist.barrier(group=group)

    def run_baseline_compute_chain() -> None:
        run_compute_chain(launch_baseline_middle)

    def run_fused_compute_chain() -> None:
        run_compute_chain(launch_fused_middle)

    def run_full(middle: Callable[[], None]) -> None:
        enqueue_dispatch()
        with torch.cuda.stream(compute_stream):
            compute_stream.wait_stream(comm_stream)
            middle()
            launch_l2_scatter()
        compute_stream.synchronize()
        dist.barrier(group=group)
        with torch.cuda.stream(compute_stream):
            launch_local_combine()
        torch.cuda.current_stream().wait_stream(compute_stream)

    def run_baseline_full() -> None:
        run_full(launch_baseline_middle)

    def run_fused_full() -> None:
        run_full(launch_fused_middle)

    # Populate the dispatch pool and compile each JIT path without an eight-rank
    # compile storm. Then execute both complete paths once on every rank.
    run_dispatch()
    torch.cuda.synchronize()
    if rank == 0:
        with torch.cuda.stream(compute_stream):
            launch_baseline_middle()
            launch_fused_middle()
            launch_l2_scatter()
            launch_local_combine()
        compute_stream.synchronize()
    dist.barrier(group=group)
    run_baseline_full()
    torch.cuda.synchronize()
    run_fused_full()
    torch.cuda.synchronize()
    dist.barrier(group=group)

    if args.check:
        if args.dispatch_scales:
            torch.testing.assert_close(sfa_pool, sfa_global, rtol=0, atol=0)
        torch.testing.assert_close(a_pool, a_global, rtol=0, atol=0)
        run_baseline_full()
        torch.cuda.synchronize()
        baseline_final = final_output.clone()
        run_fused_full()
        torch.cuda.synchronize()
        metrics = _check_final_output(final_output, baseline_final, group)
        if rank == 0:
            print(
                "check: "
                f"normalized_diff={metrics[0]:.3e}, "
                f"mean_abs={metrics[1]:.3e}, "
                f"max_abs={metrics[2]:.3e}, "
                f"mismatch_rate={metrics[3]:.3e}, "
                f"reference_abs_mean={metrics[4]:.3e}",
                flush=True,
            )
        if metrics[0] > 1.0e-5:
            raise AssertionError(
                f"end-to-end fused output normalized diff is too large: {metrics[0]:.3e}"
            )
        del baseline_final

    timings: dict[str, list[float]] = {}
    if args.components:
        timings["dispatch"] = _bench(
            run_dispatch, args.warmups, args.iters, group
        )
        timings["baseline_gather_l1_swiglu"] = _bench(
            run_baseline_middle, args.warmups, args.iters, group
        )
        timings["fused_gather_l1_swiglu"] = _bench(
            run_fused_middle, args.warmups, args.iters, group
        )
        timings["l2_scatter_and_peer_sync"] = _bench(
            run_l2_scatter, args.warmups, args.iters, group
        )
        timings["local_combine"] = _bench(
            run_local_combine, args.warmups, args.iters, group
        )
        baseline_chain, fused_chain = _bench_balanced_pair(
            run_baseline_compute_chain,
            run_fused_compute_chain,
            args.warmups,
            args.iters,
            group,
        )
        timings["baseline_l1_swiglu_l2_scatter"] = baseline_chain
        timings["fused_l1_swiglu_l2_scatter"] = fused_chain

    baseline_times, fused_times = _bench_balanced_pair(
        run_baseline_full,
        run_fused_full,
        args.warmups,
        args.iters,
        group,
    )
    timings["baseline_full"] = baseline_times
    timings["fused_full"] = fused_times

    local_medians = torch.tensor(
        [statistics.median(values) for values in timings.values()],
        device="cuda",
        dtype=torch.float64,
    )
    gathered_medians = [torch.empty_like(local_medians) for _ in range(num_ranks)]
    dist.all_gather(gathered_medians, local_medians, group=group)
    shape_ms = [None] * num_ranks
    dist.all_gather_object(shape_ms, shape_m, group=group)
    if rank == 0:
        all_medians = torch.stack(gathered_medians).cpu()
        data_bytes = a_local.numel() * a_local.element_size()
        scale_bytes = sfa_local.numel() * sfa_local.element_size() if args.dispatch_scales else 0
        print("SM90 gather-L1 + SwiGLU end-to-end MoE bench:", flush=True)
        print(
            f"  ranks={num_ranks}, tokens/rank={args.tokens_per_rank}, "
            f"M/rank={shape_ms}, E_local={local_experts}",
            flush=True,
        )
        print(
            f"  H={args.hidden}, I={args.intermediate}, top_k={args.top_k}, "
            f"scatter_dtype={args.scatter_dtype}, "
            f"swiglu_block_n={os.getenv('DG_SM90_SWIGLU_BLOCK_N', '256')}",
            flush=True,
        )
        print(
            f"  dispatch=nccl-cpp, collectives={2 if args.dispatch_scales else 1}, "
            f"payload/rank={(data_bytes + scale_bytes) / 1e6:.3f} MB, "
            f"nccl_ctas={os.environ['NCCL_MIN_CTAS']}/{os.environ['NCCL_MAX_CTAS']}",
            flush=True,
        )
        print("  rank-median latency (critical=max rank):", flush=True)
        names = list(timings)
        for index, name in enumerate(names):
            values_us = all_medians[:, index] * 1000.0
            print(
                f"    {name:30s}: median={values_us.median().item():8.2f} us, "
                f"min={values_us.min().item():8.2f} us, "
                f"max={values_us.max().item():8.2f} us",
                flush=True,
            )
        baseline_us = (all_medians[:, names.index("baseline_full")] * 1000.0).max().item()
        fused_us = (all_medians[:, names.index("fused_full")] * 1000.0).max().item()
        print(
            f"  final speedup={baseline_us / fused_us:.3f}x, "
            f"critical_saved={baseline_us - fused_us:.2f} us",
            flush=True,
        )

    dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-local-ranks", type=int, default=8)
    parser.add_argument("--tokens-per-rank", type=int, default=6976)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--top-k", type=int, default=16)
    parser.add_argument("--global-num-experts", type=int, default=512)
    parser.add_argument("--experts-per-rank-token", type=int, default=2)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--scatter-dtype", choices=("bf16", "fp8", "both"), default="both"
    )
    parser.add_argument(
        "--dispatch-scales", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--components", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    _set_default_nccl_ctas()
    if args.num_local_ranks < 1 or args.num_local_ranks > torch.cuda.device_count():
        raise ValueError("invalid num-local-ranks")
    scatter_dtypes = (
        ("bf16", "fp8") if args.scatter_dtype == "both" else (args.scatter_dtype,)
    )
    for scatter_dtype in scatter_dtypes:
        run_args = argparse.Namespace(**vars(args))
        run_args.scatter_dtype = scatter_dtype
        run_args.port = _free_port()
        print(f"Running L2 scatter dtype: {scatter_dtype}", flush=True)
        mp.spawn(
            _worker,
            args=(run_args.num_local_ranks, run_args),
            nprocs=run_args.num_local_ranks,
        )


if __name__ == "__main__":
    main()
