"""Compare gather-L1 + SwiGLU against the fused SM90 specialization."""

import argparse
import os
import socket
import statistics
import sys
from typing import Callable, List

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

try:
    import deep_gemm_moe_L1 as deep_gemm
except ModuleNotFoundError:
    import deep_gemm
    sys.modules["deep_gemm_moe_L1"] = deep_gemm

from generators import (
    MajorTypeAB,
    QuantConfig,
    cast_fp8_fp4_with_major,
    grouped_cast_fp8_fp4_with_major,
)
from sm90_gather_l1_swiglu_baseline import (
    allocate_swiglu_output,
    swiglu_quant_fp8,
)


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


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


def _bench(fn: Callable[[], None], warmups: int, iters: int,
           group: dist.ProcessGroup) -> List[float]:
    for _ in range(warmups):
        fn()
    torch.cuda.synchronize()
    dist.barrier(group)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    dist.barrier(group)
    return [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]


def _bench_balanced_pair(
    baseline_fn: Callable[[], None],
    fused_fn: Callable[[], None],
    warmups: int,
    iters: int,
    group: dist.ProcessGroup,
) -> tuple[List[float], List[float]]:
    baseline_first = _bench(baseline_fn, warmups, iters, group)
    fused_second = _bench(fused_fn, warmups, iters, group)
    fused_first = _bench(fused_fn, warmups, iters, group)
    baseline_second = _bench(baseline_fn, warmups, iters, group)
    return baseline_first + baseline_second, fused_first + fused_second


def _dequant_error_metrics(
    fused_data: torch.Tensor,
    fused_scale: torch.Tensor,
    baseline_data: torch.Tensor,
    baseline_scale: torch.Tensor,
) -> torch.Tensor:
    dot = torch.zeros((), device=fused_data.device, dtype=torch.float64)
    norm = torch.zeros_like(dot)
    abs_sum = torch.zeros_like(dot)
    max_abs = torch.zeros((), device=fused_data.device, dtype=torch.float32)
    for row_start in range(0, fused_data.shape[0], 1024):
        row_end = min(row_start + 1024, fused_data.shape[0])
        actual = fused_data[row_start:row_end].float() * fused_scale[
            row_start:row_end
        ].repeat_interleave(128, dim=1)
        expected = baseline_data[row_start:row_end].float() * baseline_scale[
            row_start:row_end
        ].repeat_interleave(128, dim=1)
        actual_d = actual.double()
        expected_d = expected.double()
        dot += (actual_d * expected_d).sum()
        norm += (actual_d.square() + expected_d.square()).sum()
        error = (actual - expected).abs()
        abs_sum += error.double().sum()
        max_abs = torch.maximum(max_abs, error.max())
    calc_diff = torch.where(norm == 0, torch.zeros_like(norm), 1.0 - 2.0 * dot / norm)
    mean_abs = abs_sum / fused_data.numel()
    return torch.stack((calc_diff, mean_abs, max_abs.double()))


def _worker(local_rank: int, num_local_ranks: int, args: argparse.Namespace) -> None:
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        "nccl",
        init_method=f"tcp://127.0.0.1:{args.port}",
        rank=local_rank,
        world_size=num_local_ranks,
        device_id=torch.device(f"cuda:{local_rank}"),
    )
    group = dist.new_group(list(range(num_local_ranks)))
    rank = dist.get_rank(group)

    if args.global_num_experts % num_local_ranks != 0:
        raise ValueError("global-num-experts must be divisible by num-local-ranks")
    if args.top_k != args.experts_per_rank_token * num_local_ranks:
        raise ValueError("top-k must equal experts-per-rank-token * num-local-ranks")
    local_experts = args.global_num_experts // num_local_ranks
    total_tokens = args.tokens_per_rank * num_local_ranks
    quant = QuantConfig()

    torch.manual_seed(0x2026)
    x_bf16 = torch.randn(
        (total_tokens, args.hidden), device="cuda", dtype=torch.bfloat16
    )
    x = cast_fp8_fp4_with_major(
        x_bf16, MajorTypeAB.KMajor, quant.gran_k_a, quant.is_fp4_a, False
    )
    del x_bf16

    routing = _make_routing(
        total_tokens, num_local_ranks, local_experts, args.experts_per_rank_token
    )
    gather_index, tile_rank, _, shape_m_tensor, psum_layout, _ = (
        deep_gemm.build_gather_layout_for_rank_overlap(
            routing,
            rank,
            num_local_ranks,
            args.tokens_per_rank,
            local_experts,
            128,
            topk_slot_offset=0,
            expert_srank_padding=False,
        )
    )
    shape_m = int(shape_m_tensor.item())
    num_tiles = (shape_m + 127) // 128
    del routing

    torch.manual_seed(0x1234 + rank)
    weight_bf16 = torch.randn(
        (local_experts, 2 * args.intermediate, args.hidden),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    weights = grouped_cast_fp8_fp4_with_major(
        weight_bf16,
        MajorTypeAB.KMajor,
        quant.gran_k_b,
        quant.is_fp4_b,
        False,
        use_block_cast_for_fp8=True,
    )
    del weight_bf16
    fused_weights = deep_gemm.transform_l1_weights_for_swiglu(weights)

    l1_output = torch.empty(
        (shape_m, 2 * args.intermediate), device="cuda", dtype=torch.bfloat16
    )
    baseline_data, baseline_scale = allocate_swiglu_output(
        shape_m, args.intermediate, l1_output.device
    )
    fused_data, fused_scale = allocate_swiglu_output(
        shape_m, args.intermediate, l1_output.device
    )
    expected_m = int((shape_m + local_experts - 1) // local_experts * 1.2)
    gemm_kwargs = dict(
        recipe=(1, 128, 128),
        disable_ue8m0_cast=True,
        use_psum_layout=True,
        expected_m_for_psum_layout=expected_m,
        gather_index=gather_index[:shape_m],
    )
    if args.use_rank_flags:
        gemm_kwargs.update(
            rank_flags=torch.ones(
                (num_local_ranks,), device="cuda", dtype=torch.int64
            ),
            tile_rank=tile_rank[:num_tiles],
            num_ranks=num_local_ranks,
            rank_flag_epoch=1,
        )

    def run_l1() -> None:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            x, weights, l1_output, psum_layout, **gemm_kwargs
        )

    def run_swiglu() -> None:
        swiglu_quant_fp8(l1_output, baseline_data, baseline_scale)

    def run_baseline() -> None:
        run_l1()
        run_swiglu()

    def run_fused() -> None:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            x,
            fused_weights,
            fused_data,
            psum_layout,
            swiglu_output_scale=fused_scale,
            **gemm_kwargs,
        )

    # Compile identical dynamic-M kernels on rank 0 before concurrent timing.
    if rank == 0:
        run_baseline()
        run_fused()
        torch.cuda.synchronize()
    dist.barrier(group)

    if args.check:
        run_baseline()
        run_fused()
        torch.cuda.synchronize()
        scale_error = (fused_scale - baseline_scale).abs().max()
        mismatch = (fused_data.float() != baseline_data.float()).float().mean()
        dequant_metrics = _dequant_error_metrics(
            fused_data, fused_scale, baseline_data, baseline_scale
        )
        check = torch.cat((torch.stack((scale_error, mismatch)).double(), dequant_metrics))
        dist.all_reduce(check, op=dist.ReduceOp.MAX, group=group)
        if rank == 0:
            print(
                f"check: max_scale_abs={check[0].item():.3e}, "
                f"max_fp8_mismatch_rate={check[1].item():.3e}, "
                f"max_dequant_diff={check[2].item():.3e}, "
                f"max_dequant_mean_abs={check[3].item():.3e}, "
                f"max_dequant_abs={check[4].item():.3e}",
                flush=True,
            )
            if check[0].item() > 1e-6 or check[1].item() > 1e-5:
                scale_diff = (fused_scale - baseline_scale).abs()
                bad_scales = scale_diff > 1e-6
                bad_data = fused_data.float() != baseline_data.float()
                scale_counts = bad_scales.sum(dim=0).cpu().tolist()
                data_counts = bad_data.reshape(
                    shape_m, args.intermediate // 128, 128
                ).sum(dim=(0, 2)).cpu().tolist()
                bad_positions = bad_scales.nonzero()[:16].cpu().tolist()
                max_position = torch.unravel_index(scale_diff.argmax(), scale_diff.shape)
                max_row, max_col = int(max_position[0]), int(max_position[1])
                baseline_dequant = baseline_data.float() * baseline_scale.repeat_interleave(128, 1)
                fused_dequant = fused_data.float() * fused_scale.repeat_interleave(128, 1)
                dequant_error = (fused_dequant - baseline_dequant).abs()
                print(f"bad scales per N128 block: {scale_counts}", flush=True)
                print(f"bad FP8 values per N128 block: {data_counts}", flush=True)
                print(f"first bad scale positions: {bad_positions}", flush=True)
                print(
                    f"max scale position=({max_row}, {max_col}), "
                    f"fused={fused_scale[max_row, max_col].item():.6e}, "
                    f"baseline={baseline_scale[max_row, max_col].item():.6e}, "
                    f"scale maxima=({fused_scale.max().item():.6e}, "
                    f"{baseline_scale.max().item():.6e})",
                    flush=True,
                )
                print(
                    f"dequant error: mean={dequant_error.mean().item():.6e}, "
                    f"max={dequant_error.max().item():.6e}, "
                    f"baseline_abs_mean={baseline_dequant.abs().mean().item():.6e}",
                    flush=True,
                )
        # Different native WGMMA N shapes can move a tiny number of values
        # across an FP8 bin boundary. Validate the compact scales tightly and
        # bound the mismatch rate instead of requiring bitwise identity.
        if (check[0].item() > 1e-6 or check[1].item() > 1e-5 or
                check[2].item() > 1e-7):
            raise AssertionError("fused output does not match the two-kernel baseline")

    gather_l1_times = _bench(run_l1, args.warmups, args.iters, group)
    swiglu_times = _bench(run_swiglu, args.warmups, args.iters, group)
    baseline_times, fused_times = _bench_balanced_pair(
        run_baseline, run_fused, args.warmups, args.iters, group
    )
    timings = {
        "gather_l1": gather_l1_times,
        "swiglu_quant": swiglu_times,
        "baseline_2_kernel": baseline_times,
        "fused_1_kernel": fused_times,
    }
    local_medians = torch.tensor(
        [statistics.median(values) for values in timings.values()],
        device="cuda",
        dtype=torch.float64,
    )
    gathered = [torch.empty_like(local_medians) for _ in range(num_local_ranks)]
    dist.all_gather(gathered, local_medians, group=group)
    if rank == 0:
        all_medians = torch.stack(gathered).cpu()
        print(
            f"shape: ranks={num_local_ranks}, M(rank0)={shape_m}, "
            f"E_local={local_experts}, H={args.hidden}, I={args.intermediate}, "
            f"rank_flags={args.use_rank_flags}",
            flush=True,
        )
        for index, name in enumerate(timings):
            values = all_medians[:, index]
            print(
                f"{name}: rank_median_us={values.median().item():.3f}, "
                f"rank_min_us={values.min().item():.3f}, "
                f"rank_max_us={values.max().item():.3f}",
                flush=True,
            )
        baseline_us = all_medians[:, 2].median().item()
        fused_us = all_medians[:, 3].median().item()
        print(
            f"speedup={baseline_us / fused_us:.3f}x, saved_us={baseline_us - fused_us:.3f}",
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
    parser.add_argument("--check", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-rank-flags", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()
    if args.num_local_ranks < 1 or args.num_local_ranks > torch.cuda.device_count():
        raise ValueError("invalid num-local-ranks")
    args.port = _free_port()
    mp.spawn(_worker, args=(args.num_local_ranks, args), nprocs=args.num_local_ranks)


if __name__ == "__main__":
    main()
