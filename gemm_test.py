import argparse
import json
import os
import re
import statistics
import sys
import time

import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "tests"))

import deep_gemm_moe_L1
from generators import (
    MajorTypeAB,
    QuantConfig,
    cast_fp8_fp4_with_major,
    grouped_cast_fp8_fp4_with_major,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-GPU reproduction for the customer allgather-overlap GEMM shape."
    )
    parser.add_argument("--profile", action="store_true",
                        help="Run timed iterations and report kernel durations.")
    parser.add_argument("--timing-mode", choices=("events", "trace"), default="events",
                        help="events: CUDA events, no CUPTI (default, recommended). "
                             "trace: torch.profiler Chrome trace (loads CUPTI; on some driver/"
                             "toolkit combos this can crash at process exit).")
    parser.add_argument("--check", action="store_true",
                        help="Check outputs before optional profiling.")
    parser.add_argument("--trace-path", type=str, default="",
                        help="Output Chrome trace path (only used when --timing-mode=trace). "
                             "Defaults to /tmp/dg_single_gemm_trace_<ts>.json.")
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--modes", nargs="+", choices=("plain", "noflags", "flags"),
                        default=("plain", "noflags", "flags"),
                        help="plain: physically gathered input; noflags: gather_index only; "
                             "flags: gather_index + rank_flags.")

    # Customer benchmark shape, mapped to local-rank GEMM inputs.
    parser.add_argument("--num-ranks", type=int, default=8)
    parser.add_argument("--tokens-per-rank", type=int, default=6976)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--n", type=int, default=2560)
    parser.add_argument("--num-experts", type=int, default=64,
                        help="Local expert count; customer global 512 experts / 8 ranks = 64.")
    parser.add_argument("--top-k", type=int, default=16,
                        help="Global top-k")
    parser.add_argument("--block-m", type=int, default=128)
    parser.add_argument("--local-rank", type=int, default=0)
    parser.add_argument("--expert-srank-padding", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Pad each expert by source rank when building the gather layout. "
                             "Default is false, which allows mixed-rank tiles and uses "
                             "tile_rank=-1 for those tiles.")
    return parser.parse_args()


def _accumulate_diff(stats: dict[str, float], x: torch.Tensor, y: torch.Tensor) -> None:
    x_d = x.double()
    y_d = y.double()
    stats["dot"] += float((x_d * y_d).sum().item())
    stats["norm"] += float((x_d * x_d + y_d * y_d).sum().item())


def _finish_diff(stats: dict[str, float]) -> float:
    if stats["norm"] == 0:
        return 0.0
    return 1.0 - 2.0 * stats["dot"] / stats["norm"]


def _compare_by_chunks(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    max_diff: float,
    rows_per_chunk: int = 512,
) -> None:
    stats = {"dot": 0.0, "norm": 0.0}
    max_abs = 0.0
    for start in range(0, actual.size(0), rows_per_chunk):
        end = min(start + rows_per_chunk, actual.size(0))
        actual_chunk = actual[start:end]
        expected_chunk = expected[start:end]
        _accumulate_diff(stats, actual_chunk, expected_chunk)
        chunk_max = float((actual_chunk.float() - expected_chunk.float()).abs().max().item())
        max_abs = max(max_abs, chunk_max)

    diff = _finish_diff(stats)
    print(f"check {name}: diff={diff:.8f}, max_abs={max_abs:.6f}, limit={max_diff}", flush=True)
    if diff >= max_diff:
        raise AssertionError(f"{name} failed: diff={diff:.8f}, limit={max_diff}")


def _check_against_reference(
    name: str,
    actual: torch.Tensor,
    a_bf16: torch.Tensor,
    b_bf16: torch.Tensor,
    psum_layout: torch.Tensor,
    max_diff: float,
) -> None:
    stats = {"dot": 0.0, "norm": 0.0}
    max_abs = 0.0
    boundaries = psum_layout.detach().cpu().tolist()

    for expert, end in enumerate(boundaries):
        start = 0 if expert == 0 else boundaries[expert - 1]
        if start == end:
            continue
        ref = (a_bf16[start:end].float() @ b_bf16[expert].float().t()).to(torch.bfloat16)
        actual_slice = actual[start:end]
        _accumulate_diff(stats, actual_slice, ref)
        chunk_max = float((actual_slice.float() - ref.float()).abs().max().item())
        max_abs = max(max_abs, chunk_max)
        del ref

    diff = _finish_diff(stats)
    print(f"check {name}: diff={diff:.8f}, max_abs={max_abs:.6f}, limit={max_diff}", flush=True)
    if diff >= max_diff:
        raise AssertionError(f"{name} failed: diff={diff:.8f}, limit={max_diff}")


def summarize_trace(trace_path: str) -> None:
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]

    label_re = re.compile(r"^single_(plain|noflags|flags)/iter_\d+$")
    labels = [
        event for event in events
        if event.get("cat") == "gpu_user_annotation" and label_re.match(event.get("name", ""))
    ]
    kernels = [
        event for event in events
        if event.get("cat") == "kernel" and "sm90_fp8_gemm_1d2d_impl" in event.get("name", "")
    ]

    by_label = {"single_plain": [], "single_noflags": [], "single_flags": []}
    for kernel in sorted(kernels, key=lambda event: event["ts"]):
        mid_ts = kernel["ts"] + kernel.get("dur", 0.0) / 2
        matches = [
            label for label in labels
            if label["ts"] <= mid_ts <= label["ts"] + label.get("dur", 0.0)
        ]
        if not matches:
            continue
        label = min(matches, key=lambda event: event.get("dur", 0.0))
        by_label[label["name"].split("/")[0]].append(float(kernel["dur"]))

    print(f"trace={trace_path}", flush=True)
    for label, values in by_label.items():
        if not values:
            print(f"{label}: no sm90_fp8_gemm_1d2d_impl kernels matched", flush=True)
            continue
        print(
            f"{label}: n={len(values)} "
            f"median_us={statistics.median(values):.3f} "
            f"mean_us={statistics.mean(values):.3f} "
            f"min_us={min(values):.3f} "
            f"max_us={max(values):.3f}",
            flush=True,
        )


def main() -> None:
    args = parse_args()

    total_tokens = args.num_ranks * args.tokens_per_rank
    local_num_experts = args.num_experts
    global_num_experts = args.num_ranks * args.num_experts
    quant_config = QuantConfig()
    recipe, recipe_a, recipe_b = quant_config.get_recipes()

    torch.manual_seed(0x2026)
    a_global_bf16 = torch.randn(
        (total_tokens, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    a_data, sfa = cast_fp8_fp4_with_major(
        a_global_bf16,
        MajorTypeAB.KMajor,
        quant_config.gran_k_a,
        quant_config.is_fp4_a,
        use_ue8m0=False,
    )

    gen = torch.Generator(device="cuda").manual_seed(0xBEEF)
    scores = torch.rand(
        (total_tokens, global_num_experts),
        dtype=torch.float32,
        device="cuda",
        generator=gen,
    )
    routing_topk = torch.topk(scores, args.top_k, dim=1).indices.to(torch.int32)

    gather_index, tile_rank, grouped_layout, m_logical_t, psum_layout, _row_to_topk = (
        deep_gemm_moe_L1.build_gather_layout_for_rank_overlap(
            routing_topk,
            args.local_rank,
            args.num_ranks,
            args.tokens_per_rank,
            local_num_experts,
            args.block_m,
            0,
            args.expert_srank_padding,
        )
    )
    m_logical = int(m_logical_t.item())
    num_tiles = (m_logical + args.block_m - 1) // args.block_m
    print(
        f"expert_srank_padding={args.expert_srank_padding}, "
        f"m_logical={m_logical}, num_tiles={num_tiles}",
        flush=True,
    )

    torch.manual_seed(0x5678)
    b_bf16 = torch.randn(
        (local_num_experts, args.n, args.hidden),
        dtype=torch.bfloat16,
        device="cuda",
    )
    b = grouped_cast_fp8_fp4_with_major(
        b_bf16,
        MajorTypeAB.KMajor,
        quant_config.gran_k_b,
        quant_config.is_fp4_b,
        use_ue8m0=False,
        use_block_cast_for_fp8=True,
    )

    d = torch.empty((m_logical, args.n), dtype=torch.bfloat16, device="cuda")
    rank_flags = torch.ones((args.num_ranks,), dtype=torch.int64, device="cuda")
    expected_m_per_expert = int((m_logical + local_num_experts - 1) // local_num_experts * 1.2)

    needs_plain = args.check or "plain" in args.modes
    is_pad = None
    a_logical_bf16 = None
    a_logical = None
    if needs_plain:
        print("building physically gathered plain GEMM input", flush=True)
        gi = gather_index[:m_logical].long()
        is_pad = gi < 0
        safe_gi = torch.where(is_pad, torch.zeros_like(gi), gi)
        a_logical_bf16 = a_global_bf16[safe_gi].clone()
        a_logical_bf16[is_pad] = 0
        a_logical = cast_fp8_fp4_with_major(
            a_logical_bf16,
            MajorTypeAB.KMajor,
            quant_config.gran_k_a,
            quant_config.is_fp4_a,
            use_ue8m0=False,
        )

    def launch(mode: str, output: torch.Tensor) -> None:
        if mode == "plain":
            if a_logical is None:
                raise RuntimeError("plain mode requested but plain input was not built")
            launch_plain(a_logical, output)
            return

        kwargs = {}
        if mode == "flags":
            kwargs.update(
                rank_flags=rank_flags,
                tile_rank=tile_rank[:num_tiles],
                num_ranks=args.num_ranks,
                rank_flag_epoch=1,
            )
        deep_gemm_moe_L1.m_grouped_fp8_gemm_nt_contiguous(
            (a_data, sfa),
            b,
            output,
            psum_layout,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            disable_ue8m0_cast=True,
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m_per_expert,
            gather_index=gather_index[:m_logical],
            **kwargs,
        )

    def launch_plain(a_plain, output: torch.Tensor) -> None:
        deep_gemm_moe_L1.m_grouped_fp8_gemm_nt_contiguous(
            a_plain,
            b,
            output,
            psum_layout,
            recipe=recipe,
            recipe_a=recipe_a,
            recipe_b=recipe_b,
            disable_ue8m0_cast=True,
            use_psum_layout=True,
            expected_m_for_psum_layout=expected_m_per_expert,
        )

    if args.check:
        d_plain = torch.empty_like(d)
        launch_plain(a_logical, d_plain)
        torch.cuda.synchronize()
        _check_against_reference(
            "plain_vs_bf16_ref",
            d_plain,
            a_logical_bf16,
            b_bf16,
            psum_layout,
            quant_config.max_diff(),
        )

        d_test = torch.empty_like(d)
        for mode in args.modes:
            launch(mode, d_test)
            torch.cuda.synchronize()
            _compare_by_chunks(f"{mode}_vs_plain", d_test, d_plain, max_diff=1e-6)
            if bool(is_pad.any().item()):
                pad_max = float(d_test[is_pad].float().abs().max().item())
                print(f"check {mode}_pad_rows: max_abs={pad_max:.6f}", flush=True)
                if pad_max != 0.0:
                    raise AssertionError(f"{mode} pad rows are not zero: max_abs={pad_max}")
        print("check: passed", flush=True)

    if not args.profile:
        for mode in args.modes:
            print(f"launch_once mode={mode}", flush=True)
            launch(mode, d)
        torch.cuda.synchronize()
        return

    for _ in range(args.warmups):
        for mode in args.modes:
            launch(mode, d)
        torch.cuda.synchronize()

    if args.timing_mode == "events":
        # CUDA events path: no CUPTI, no torch.profiler. Each iter brackets a
        # single launch() with start/end events on the current stream. We
        # synchronize once at the end and convert ms->us.
        for mode in args.modes:
            starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
            ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.iters)]
            for i in range(args.iters):
                starts[i].record()
                launch(mode, d)
                ends[i].record()
            torch.cuda.synchronize()
            durations_us = [s.elapsed_time(e) * 1000.0 for s, e in zip(starts, ends)]
            print(
                f"single_{mode}: n={len(durations_us)} "
                f"median_us={statistics.median(durations_us):.3f} "
                f"mean_us={statistics.mean(durations_us):.3f} "
                f"min_us={min(durations_us):.3f} "
                f"max_us={max(durations_us):.3f}",
                flush=True,
            )
        return

    trace_path = args.trace_path
    if not trace_path:
        trace_path = f"/tmp/dg_single_gemm_trace_{int(time.time())}.json"
    os.makedirs(os.path.dirname(os.path.abspath(trace_path)), exist_ok=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
    ) as prof:
        for mode in args.modes:
            label = f"single_{mode}"
            for i in range(args.iters):
                torch.cuda.synchronize()
                with torch.profiler.record_function(f"{label}/iter_{i}"):
                    launch(mode, d)
                torch.cuda.synchronize()

    prof.export_chrome_trace(trace_path)
    summarize_trace(trace_path)


if __name__ == "__main__":
    main()
