#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm {

// =============================================================================
// MoE gather-layout generator runtimes
//
// Four small kernels build (gather_index, tile_rank, grouped_layout, m_logical,
// row_to_topk) from a global routing-topk tensor for the per-rank-flag overlap
// GEMM path.
// See `docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md` (§11) for the full
// design rationale.
//
// Pipeline:
//   Phase 1  histogram          (multi-block, per token)
//   Phase 2a prefix              (single block, header only)
//   Phase 2b fill_layout_tables  (multi-block, 1 block per chunk → fills
//                                 grouped_layout + tile_rank)
//   Phase 3  scatter             (multi-block, per token)
//
// Phase 2 used to be a single block that also filled grouped_layout for every
// output row, which was the absolute bottleneck (single-SM HBM throughput on
// `m_logical ≈ 1 M ints`). Splitting into 2a + 2b moves the hot fill onto
// `num_experts * num_ranks` parallel blocks.
//
// All kernel sources live in `deep_gemm/include/deep_gemm/impls/moe_gather_layout.cuh`
// and are JIT-compiled with `kNumThreads` as the only compile-time parameter.
// =============================================================================


// Phase 1: histogram counts[e][r] from (token, expert) pairs.
class GatherLayoutHistogramRuntime final : public LaunchRuntime<GatherLayoutHistogramRuntime> {
public:
    struct Args {
        void *routing_topk, *counts;
        uint32_t T, K, num_experts, num_ranks, tokens_per_rank;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/moe_gather_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&histogram_for_gather_layout<{}>);
}};
)", args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.routing_topk, args.counts,
            args.T, args.K, args.num_experts, args.num_ranks, args.tokens_per_rank));
    }
};


// Phase 2a: serial prefix sum + m_logical (single block).
class GatherLayoutPrefixRuntime final : public LaunchRuntime<GatherLayoutPrefixRuntime> {
public:
    struct Args {
        void *counts, *padded_starts, *m_logical_out;
        uint32_t local_rank, num_experts, num_ranks, block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/moe_gather_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&prefix_for_gather_layout<{}>);
}};
)", args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.counts, args.padded_starts, args.m_logical_out,
            args.local_rank, args.num_experts, args.num_ranks, args.block_m));
    }
};


// Phase 2b: per-chunk grouped_layout + tile_rank fill (multi-block).
class GatherLayoutFillTablesRuntime final : public LaunchRuntime<GatherLayoutFillTablesRuntime> {
public:
    struct Args {
        void *counts, *padded_starts, *tile_rank, *grouped_layout;
        uint32_t local_rank, num_experts, num_ranks, block_m;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/moe_gather_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&fill_layout_tables_for_gather_layout<{}>);
}};
)", args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.counts, args.padded_starts, args.tile_rank, args.grouped_layout,
            args.local_rank, args.num_experts, args.num_ranks, args.block_m));
    }
};


// Phase 3: scatter gather_index[pos] = token_id and row_to_topk[pos] = slot.
class GatherLayoutScatterRuntime final : public LaunchRuntime<GatherLayoutScatterRuntime> {
public:
    struct Args {
        void *routing_topk, *padded_starts, *chunk_cursor, *gather_index, *row_to_topk;
        uint32_t T, K, local_rank, num_experts, num_ranks, tokens_per_rank, topk_slot_offset;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/moe_gather_layout.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&scatter_for_gather_layout<{}>);
}};
)", args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.routing_topk, args.padded_starts, args.chunk_cursor, args.gather_index, args.row_to_topk,
            args.T, args.K, args.local_rank, args.num_experts, args.num_ranks,
            args.tokens_per_rank, args.topk_slot_offset));
    }
};


// -----------------------------------------------------------------------------
// Host entry: build_gather_layout_for_rank_overlap
//
// Returns six tensors needed by the rank-overlap GEMM path:
//   gather_index:      (M_max,)           int32, pad rows = -1
//   tile_rank:         (num_m_tiles_max,) int32
//   grouped_layout:    (M_max,)           int32, per-row expert id (for psum=0)
//   m_logical_tensor:  (1,)               int32, the actual padded M
//   psum_layout:       (num_experts,)     int32, cumulative M boundary per expert
//                      (for use_psum_layout=True — pass as grouped_layout to GEMM)
//   row_to_topk:       (M_max,)           int32, pad rows = -1; logical row →
//                      top-k output slot for combine-scatter experiments
//
// `M_max` is an analytical upper bound (tight enough that real workloads see
// ~70% utilization on the H800 reference shape — vs. ~3.5% under the previous
// `num_experts * num_ranks * tokens_per_rank` bound):
//
//     M_logical ≤ Σ_chunks n_real + Σ_chunks n_pad
//               ≤ T * K              + min(num_chunks, T*K) * (block_m - 1)
//
// where T*K = num_ranks * tokens_per_rank * top_k is the exact upper bound on
// total non-pad output rows (each token contributes top_k entries; OOB expert
// ids are dropped → the bound stays an upper bound), and at most
// `min(num_chunks, T*K)` chunks can be non-empty (each non-empty chunk
// contributes ≤ block_m-1 pad rows).
//
// Caller may pass `gather_index` / `grouped_layout` directly to the GEMM, which
// reads only the first `m_logical` rows; `tile_rank` is similarly over-
// allocated and only the first `ceil(m_logical / block_m)` entries matter.
// For the psum path, pass `psum_layout` as the grouped_layout argument with
// `use_psum_layout=True`.
// -----------------------------------------------------------------------------
static std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
build_gather_layout_for_rank_overlap(
    const torch::Tensor& routing_topk,
    const int& local_rank,
    const int& num_ranks,
    const int& tokens_per_rank,
    const int& num_experts,
    const int& block_m,
    const int& topk_slot_offset = 0)
{
    // Input checks
    DG_HOST_ASSERT(routing_topk.dim() == 2);
    DG_HOST_ASSERT(routing_topk.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(routing_topk.is_cuda() and routing_topk.is_contiguous());
    DG_HOST_ASSERT(num_ranks > 0 and num_ranks <= 8);
    DG_HOST_ASSERT(local_rank >= 0 and local_rank < num_ranks);
    DG_HOST_ASSERT(num_experts > 0);
    DG_HOST_ASSERT(tokens_per_rank > 0);
    DG_HOST_ASSERT(block_m > 0);
    DG_HOST_ASSERT(topk_slot_offset >= 0);

    const int T = static_cast<int>(routing_topk.size(0));
    const int K = static_cast<int>(routing_topk.size(1));
    DG_HOST_ASSERT(T == num_ranks * tokens_per_rank);
    DG_HOST_ASSERT(K > 0);

    // Allocate outputs and aux buffers on the input device.
    const auto opts_int32 = at::TensorOptions().dtype(torch::kInt).device(routing_topk.device());

    // Tighter analytical upper bound on M_logical: see the section comment
    // above and `docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md` §11.4.4.
    // We keep `int64_t` for the intermediate to defend against `T * K` flowing
    // past `int32` on absurdly large configs (it cannot under our current
    // public API limits, but the cost of the wider math is nil).
    const int64_t total_pairs = static_cast<int64_t>(num_ranks) *
                                static_cast<int64_t>(tokens_per_rank) *
                                static_cast<int64_t>(K);
    const int64_t num_chunks = static_cast<int64_t>(num_experts) *
                               static_cast<int64_t>(num_ranks);
    const int64_t max_nonempty_chunks = std::min<int64_t>(num_chunks, total_pairs);
    const int64_t M_max64 = total_pairs + max_nonempty_chunks * (block_m - 1);
    DG_HOST_ASSERT(M_max64 < static_cast<int64_t>(std::numeric_limits<int>::max()));
    const int M_max = static_cast<int>(M_max64);
    // tile_rank covers up to `ceil(M_max / block_m)` tiles; the +1 ceil is
    // defensive since `M_max` need not be a multiple of `block_m` (the bound
    // adds `block_m - 1` slack per non-empty chunk).
    const int num_m_tiles_max = ceil_div(M_max, block_m);

    // Outputs: gather_index/row_to_topk pre-filled with -1 so that any position
    // not written by Phase 3 (scatter) automatically reads as a pad row.
    auto gather_index = torch::full({M_max}, -1, opts_int32);
    auto row_to_topk = torch::full({M_max}, -1, opts_int32);
    auto tile_rank = torch::empty({num_m_tiles_max}, opts_int32);
    // grouped_layout is fully (re)written by Phase 2b; no need to pre-init.
    auto grouped_layout = torch::empty({M_max}, opts_int32);
    auto m_logical_tensor = torch::empty({1}, opts_int32);

    // Aux buffers: counts and chunk_cursor must be zero-initialized.
    auto counts = torch::zeros({num_experts, num_ranks}, opts_int32);
    auto padded_starts = torch::empty({num_experts, num_ranks}, opts_int32);
    auto chunk_cursor = torch::zeros({num_experts, num_ranks}, opts_int32);

    // ---------- Phase 1: histogram ----------
    {
        constexpr int num_threads = 256;
        const int num_blocks = ceil_div(T, num_threads);
        const auto args = GatherLayoutHistogramRuntime::Args{
            .routing_topk = routing_topk.data_ptr(),
            .counts = counts.data_ptr(),
            .T = static_cast<uint32_t>(T),
            .K = static_cast<uint32_t>(K),
            .num_experts = static_cast<uint32_t>(num_experts),
            .num_ranks = static_cast<uint32_t>(num_ranks),
            .tokens_per_rank = static_cast<uint32_t>(tokens_per_rank),
            .launch_args = LaunchArgs(num_blocks, num_threads),
        };
        const auto code = GatherLayoutHistogramRuntime::generate(args);
        const auto runtime = compiler->build("moe_gather_layout_histogram", code);
        GatherLayoutHistogramRuntime::launch(runtime, args);
    }

    // ---------- Phase 2a: block-wide parallel prefix sum (single block) -----
    {
        // Single block doing a warp-shuffle block scan over `n_slot[*]`. We
        // use 256 threads = 8 warps which can host up to
        // `kNumThreads * kChunksPerThread = 256 * 16 = 4096` chunks (the
        // largest configuration we currently care about: 512 experts × 8
        // ranks).
        //
        // SMEM holds one int32 array of length `num_experts * num_ranks`
        // (`s_counts`) staged from gmem; `padded_starts` is written directly
        // to global memory by every thread.
        constexpr int num_threads = 256;
        const int smem_size = num_experts * num_ranks * static_cast<int>(sizeof(int));
        const auto args = GatherLayoutPrefixRuntime::Args{
            .counts = counts.data_ptr(),
            .padded_starts = padded_starts.data_ptr(),
            .m_logical_out = m_logical_tensor.data_ptr(),
            .local_rank = static_cast<uint32_t>(local_rank),
            .num_experts = static_cast<uint32_t>(num_experts),
            .num_ranks = static_cast<uint32_t>(num_ranks),
            .block_m = static_cast<uint32_t>(block_m),
            .launch_args = LaunchArgs(1, num_threads, smem_size),
        };
        const auto code = GatherLayoutPrefixRuntime::generate(args);
        const auto runtime = compiler->build("moe_gather_layout_prefix", code);
        GatherLayoutPrefixRuntime::launch(runtime, args);
    }

    // ---------- Phase 2b: per-chunk grouped_layout + tile_rank fill ----------
    {
        // 1 block per (e, s) chunk. 128 threads is enough — the inner loop is
        // pure global stores (`grouped_layout[start + i] = e`) and most chunks
        // are 1–2 tiles deep, so adding more threads doesn't speed up the
        // single chunk noticeably. Total grid is `num_experts * num_ranks ≤
        // 4096` blocks for the 512-experts × 8-ranks reference shape; on a
        // 132-SM H800 that's ~32 waves with each wave essentially HBM-bound.
        constexpr int num_threads = 128;
        const auto args = GatherLayoutFillTablesRuntime::Args{
            .counts = counts.data_ptr(),
            .padded_starts = padded_starts.data_ptr(),
            .tile_rank = tile_rank.data_ptr(),
            .grouped_layout = grouped_layout.data_ptr(),
            .local_rank = static_cast<uint32_t>(local_rank),
            .num_experts = static_cast<uint32_t>(num_experts),
            .num_ranks = static_cast<uint32_t>(num_ranks),
            .block_m = static_cast<uint32_t>(block_m),
            .launch_args = LaunchArgs({num_experts, num_ranks}, num_threads),
        };
        const auto code = GatherLayoutFillTablesRuntime::generate(args);
        const auto runtime = compiler->build("moe_gather_layout_fill_tables", code);
        GatherLayoutFillTablesRuntime::launch(runtime, args);
    }

    // ---------- Phase 3: scatter ----------
    {
        constexpr int num_threads = 256;
        const int num_blocks = ceil_div(T, num_threads);
        const auto args = GatherLayoutScatterRuntime::Args{
            .routing_topk = routing_topk.data_ptr(),
            .padded_starts = padded_starts.data_ptr(),
            .chunk_cursor = chunk_cursor.data_ptr(),
            .gather_index = gather_index.data_ptr(),
            .row_to_topk = row_to_topk.data_ptr(),
            .T = static_cast<uint32_t>(T),
            .K = static_cast<uint32_t>(K),
            .local_rank = static_cast<uint32_t>(local_rank),
            .num_experts = static_cast<uint32_t>(num_experts),
            .num_ranks = static_cast<uint32_t>(num_ranks),
            .tokens_per_rank = static_cast<uint32_t>(tokens_per_rank),
            .topk_slot_offset = static_cast<uint32_t>(topk_slot_offset),
            .launch_args = LaunchArgs(num_blocks, num_threads),
        };
        const auto code = GatherLayoutScatterRuntime::generate(args);
        const auto runtime = compiler->build("moe_gather_layout_scatter", code);
        GatherLayoutScatterRuntime::launch(runtime, args);
    }

    // ---------- Derive psum_layout from padded_starts + m_logical ----------
    // psum_layout[e] = cumulative M boundary after expert e.
    // Since the layout is expert-major (ring-step-minor within each expert),
    // the end of expert e = start of expert (e+1)'s first chunk, which is
    // padded_starts[e+1, 0]. The last expert ends at m_logical.
    auto psum_layout = torch::empty({num_experts}, opts_int32);
    if (num_experts > 1) {
        psum_layout.slice(0, 0, num_experts - 1).copy_(
            padded_starts.slice(0, 1, num_experts).select(1, 0));
    }
    psum_layout.slice(0, num_experts - 1, num_experts).copy_(m_logical_tensor);

    return std::make_tuple(gather_index, tile_rank, grouped_layout, m_logical_tensor, psum_layout, row_to_topk);
}

}  // namespace deep_gemm
