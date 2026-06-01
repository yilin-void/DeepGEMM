#pragma once

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

// =============================================================================
// MoE gather-layout generator kernels
//
// Builds (gather_index, tile_rank, grouped_layout, m_logical) from a global
// MoE routing map for the per-rank-flag overlap path
// (see docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md).
//
// Layout A (expert-major outer + ring-order rank-minor inner):
//
//   start = 0
//   for e in [0, num_experts):
//       for s in [0, num_ranks):
//           rank_id = (local_rank + s) % num_ranks
//           n_real = count[e][rank_id]
//           n_slot = ceil_div(n_real, block_m) * block_m
//           [start, start + n_real)         <- real source rows
//           [start + n_real, start + n_slot) <- pad rows (gather_index = -1)
//           grouped_layout[start ... start + n_slot) = e
//           tile_rank for the (n_slot/block_m) tiles in this chunk = rank_id
//           start += n_slot
//   m_logical = start
//
// Indexing conventions used across the four phases:
//   counts[e][r]         : e-major, raw rank id (NOT ring step)
//   padded_starts[e][s]  : e-major, ring step inner
//   chunk_cursor[e][s]   : e-major, ring step inner
//
// Pipeline overview:
//   Phase 1 (multi-block): counts[e][r] = histogram(routing_topk)
//   Phase 2a (single block): padded_starts[e][s] = prefix_sum(n_slot[e][s])
//                            and m_logical_out
//   Phase 2b (multi-block, grid=(num_experts, num_ranks)):
//                            grouped_layout + tile_rank fill (one block per
//                            chunk; replaces the old single-block fill that
//                            dominated total runtime)
//   Phase 3 (multi-block): scatter gather_index using padded_starts
// =============================================================================


// -----------------------------------------------------------------------------
// Phase 1 — Histogram: counts[e][r] += 1 for each (token, expert) pair
// -----------------------------------------------------------------------------
template <uint32_t kNumThreads>
__global__ void histogram_for_gather_layout(
    const int* __restrict__ routing_topk,   // (T, K), values are GLOBAL expert ids
    int* __restrict__ counts,               // (num_experts, num_ranks), zero-init by host
    uint32_t T,
    uint32_t K,
    uint32_t num_experts,
    uint32_t num_ranks,
    uint32_t tokens_per_rank,
    uint32_t expert_base)                   // global id of this rank's first local expert
{
    const uint32_t t = blockIdx.x * kNumThreads + threadIdx.x;
    if (t >= T) return;

    const uint32_t r = t / tokens_per_rank;     // rank that owns this token

    #pragma unroll 1
    for (uint32_t j = 0; j < K; ++j) {
        const int g = __ldg(routing_topk + t * K + j);   // global expert id
        // Map global id -> this rank's local id and keep only experts owned by
        // this rank. Negative ids (e.g. -1 "no expert") and experts on other
        // ranks are silently dropped; this also guards against OOB writes.
        const uint32_t e = static_cast<uint32_t>(g - static_cast<int>(expert_base));
        if (g >= 0 && e < num_experts)
            atomicAdd(&counts[e * num_ranks + r], 1);
    }
}


// -----------------------------------------------------------------------------
// Phase 2a — Block-wide parallel prefix sum (single block)
//
// Computes padded_starts[e][s] from counts[e][r] using the layout-A traversal
// order (expert-major outer, ring-step inner). Writes m_logical_out.
//
// **Does NOT fill tile_rank or grouped_layout** — that work is delegated to
// Phase 2b which runs one block per chunk in parallel across SMs.
//
// Implementation note: a block-wide exclusive scan over `n_slot[*]` is used
// (instead of an obvious thread-0 serial loop) because the serial version is
// surprisingly expensive (~300 µs at 4096 chunks on H800): each iteration
// pays for at least two integer divisions (modulo by `num_ranks`, integer
// divide by `block_m`) and the loop trip is sequential. The parallel scan
// distributes the divisions across all 256 threads and combines per-warp
// results via warp-shuffle, dropping Phase 2a to single-digit µs.
//
// Algorithm:
//   1. Cooperatively load `counts` into smem.
//   2. Each thread computes `n_slot` for `kChunksPerThread` contiguous
//      chunks in [e][s]-linear order, accumulating a per-thread total.
//   3. Block-wide exclusive scan over per-thread totals (warp-shuffle scan
//      + smem combine across warps), giving each thread its global prefix.
//   4. Each thread sequentially writes its `padded_starts` slice and adds
//      `n_slot` to the running prefix.
//   5. Thread 0 publishes `m_logical_out`.
//
// SMEM layout (single dynamic allocation):
//   [0,             total_chunks): s_counts       (staged from gmem)
//   [total_chunks,  2*total_chunks): scratch (unused; kept aligned for the
//      historical contract — host still allocates `2 * total_chunks * 4 B`)
//
// `kChunksPerThread` is a compile-time constant. Configurations with
// `num_experts * num_ranks > kNumThreads * kChunksPerThread` will trip the
// device-side assert; the host bumps either kNumThreads or this constant
// before launching such a configuration. With kNumThreads=256 and
// kChunksPerThread=16, the kernel handles up to 4096 chunks (= 512 experts ×
// 8 ranks), which is the upper end of all current MoE configs.
// -----------------------------------------------------------------------------
template <uint32_t kNumThreads>
__global__ void prefix_for_gather_layout(
    const int* __restrict__ counts,         // (num_experts, num_ranks)
    int* __restrict__ padded_starts,        // (num_experts, num_ranks) - indexed by [e][s]
    int* __restrict__ m_logical_out,        // scalar
    uint32_t local_rank,
    uint32_t num_experts,
    uint32_t num_ranks,
    uint32_t block_m)
{
    static_assert(kNumThreads % 32 == 0, "block must contain whole warps");
    constexpr uint32_t kNumWarps = kNumThreads / 32;
    static_assert(kNumWarps <= 32, "warp-of-warp scan needs <= 32 warps");
    constexpr uint32_t kChunksPerThread = 16;
    constexpr uint32_t kMaxChunks = kNumThreads * kChunksPerThread;

    extern __shared__ int s_buf[];
    const uint32_t total_chunks = num_experts * num_ranks;
    DG_TRAP_ONLY_DEVICE_ASSERT(total_chunks <= kMaxChunks);
    int* s_counts = s_buf;

    // Step 1: cooperative load of counts → smem (coalesced + parallel).
    for (uint32_t i = threadIdx.x; i < total_chunks; i += kNumThreads)
        s_counts[i] = counts[i];
    __syncthreads();

    // Step 2: each thread computes `n_slot` for its blocked range of chunks
    // (block layout: thread tid handles [tid * kChunksPerThread, ...) ).
    const uint32_t base = threadIdx.x * kChunksPerThread;
    int local_n_slot[kChunksPerThread];
    int local_total = 0;
    #pragma unroll
    for (uint32_t i = 0; i < kChunksPerThread; ++i) {
        const uint32_t idx = base + i;
        int n_slot = 0;
        if (idx < total_chunks) {
            const uint32_t e = idx / num_ranks;
            const uint32_t s = idx - e * num_ranks;             // == idx % num_ranks
            const uint32_t r = (local_rank + s) % num_ranks;
            const int n_real = s_counts[e * num_ranks + r];
            n_slot = ((n_real + static_cast<int>(block_m) - 1)
                      / static_cast<int>(block_m))
                     * static_cast<int>(block_m);
        }
        local_n_slot[i] = n_slot;
        local_total += n_slot;
    }

    // Step 3: block-wide exclusive scan over `local_total`.
    //   3a: intra-warp inclusive scan via shfl_up.
    const uint32_t lane = threadIdx.x & 31;
    const uint32_t warp_id = threadIdx.x >> 5;
    int warp_inc = local_total;
    #pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int v = __shfl_up_sync(0xffffffffu, warp_inc, o);
        if (lane >= static_cast<uint32_t>(o)) warp_inc += v;
    }

    //   3b: per-warp totals → smem; one warp scans those totals; broadcast.
    __shared__ int s_warp_excl[kNumWarps];
    __shared__ int s_block_total;
    if (lane == 31u)
        s_warp_excl[warp_id] = warp_inc;
    __syncthreads();

    if (warp_id == 0u) {
        int v = (lane < kNumWarps) ? s_warp_excl[lane] : 0;
        int incl = v;
        #pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int u = __shfl_up_sync(0xffffffffu, incl, o);
            if (lane >= static_cast<uint32_t>(o)) incl += u;
        }
        if (lane < kNumWarps)
            s_warp_excl[lane] = incl - v;                       // exclusive per-warp prefix
        if (lane == kNumWarps - 1u)
            s_block_total = incl;                               // grand total
    }
    __syncthreads();

    // Each thread's exclusive block prefix = warp's exclusive prefix
    //                                       + (warp inclusive scan up to lane) - local_total.
    const int my_excl_prefix = s_warp_excl[warp_id] + (warp_inc - local_total);

    // Step 4: write padded_starts in [e][s] order. The chunk traversal in
    // step 2 already produced n_slot in linear-idx order, so writes here go
    // to the same indices.
    int prefix = my_excl_prefix;
    #pragma unroll
    for (uint32_t i = 0; i < kChunksPerThread; ++i) {
        const uint32_t idx = base + i;
        if (idx < total_chunks) {
            padded_starts[idx] = prefix;
            prefix += local_n_slot[i];
        }
    }

    // Step 5: publish m_logical.
    if (threadIdx.x == 0)
        *m_logical_out = s_block_total;
}


// -----------------------------------------------------------------------------
// Phase 2b — Per-chunk grouped_layout + tile_rank fill (multi-block).
//
// Grid: (num_experts, num_ranks). Each block owns exactly one (e, s) chunk and:
//   * fills grouped_layout[start .. start + n_slot) with `e` (real + pad alike;
//     Phase 3 doesn't touch grouped_layout at all, so this is the sole writer)
//   * fills tile_rank[start/block_m .. start/block_m + n_tiles) with `r`.
//
// Why a 2D grid: launching `num_experts * num_ranks ≤ 64*8 = 512` (or 512*8 =
// 4096 for the very-many-experts setup) blocks lets ~all SMs participate in
// parallel; each block's `n_slot` write is at most `tokens_per_rank` rows so
// the entire phase becomes HBM-throughput bound (single-digit ms → tens of µs
// in our reference shape).
// -----------------------------------------------------------------------------
template <uint32_t kNumThreads>
__global__ void fill_layout_tables_for_gather_layout(
    const int* __restrict__ counts,         // (num_experts, num_ranks)
    const int* __restrict__ padded_starts,  // (num_experts, num_ranks) - [e][s]
    int* __restrict__ tile_rank,            // (num_m_tiles_max,)
    int* __restrict__ grouped_layout,       // (M_logical_max,)
    uint32_t local_rank,
    uint32_t num_experts,
    uint32_t num_ranks,
    uint32_t block_m)
{
    const uint32_t e = blockIdx.x;
    const uint32_t s = blockIdx.y;
    if (e >= num_experts or s >= num_ranks) return;

    const uint32_t r = (local_rank + s) % num_ranks;
    const int n_real = counts[e * num_ranks + r];
    const int n_slot = ((n_real + static_cast<int>(block_m) - 1)
                        / static_cast<int>(block_m))
                       * static_cast<int>(block_m);
    if (n_slot == 0)
        return;                                  // empty chunk: no rows / no tiles

    const int start = padded_starts[e * num_ranks + s];

    // Fill grouped_layout: one int32 per output row, all = e. The hot loop is
    // pure global stores; coalesced access (consecutive threads → consecutive
    // bytes in `start + i`).
    for (int i = threadIdx.x; i < n_slot; i += kNumThreads)
        grouped_layout[start + i] = static_cast<int>(e);

    // Fill tile_rank: one int32 per m-tile, all = r. n_tiles is small
    // (≤ tokens_per_rank/block_m ≈ 58 for our reference shape) so a few
    // threads do all the writes; the rest no-op (their `t` exceeds n_tiles).
    const int n_tiles = n_slot / static_cast<int>(block_m);
    const int tile_base = start / static_cast<int>(block_m);
    for (int t = threadIdx.x; t < n_tiles; t += kNumThreads)
        tile_rank[tile_base + t] = static_cast<int>(r);
}


// -----------------------------------------------------------------------------
// Phase 3 — Scatter: for each (token, expert_choice), atomically take a slot
// in the (e, s) chunk and write gather_index[pos] = token_id and
// row_to_topk[pos] = top-k slot.
//
// gather_index and row_to_topk are host-pre-initialized to -1, so positions not
// written by this pass remain `-1` and are interpreted as "pad row" by the GEMM
// kernel (see §11.4.2). grouped_layout for those same positions is pre-filled
// to the chunk's expert id by Phase 2b (so pad rows still produce 0·B[expert] =
// 0 rather than getting masked out).
// -----------------------------------------------------------------------------
template <uint32_t kNumThreads>
__global__ void scatter_for_gather_layout(
    const int* __restrict__ routing_topk,    // (T, K), values are GLOBAL expert ids
    const int* __restrict__ padded_starts,   // (num_experts, num_ranks) - [e][s]
    int* __restrict__ chunk_cursor,          // (num_experts, num_ranks), zero-init by host
    int* __restrict__ gather_index,          // (M_logical_max,) pre-init = -1
    int* __restrict__ row_to_topk,           // (M_logical_max,) pre-init = -1
    uint32_t T,
    uint32_t K,
    uint32_t local_rank,
    uint32_t num_experts,
    uint32_t num_ranks,
    uint32_t tokens_per_rank,
    uint32_t topk_slot_offset,
    uint32_t expert_base)                    // global id of this rank's first local expert
{
    const uint32_t t = blockIdx.x * kNumThreads + threadIdx.x;
    if (t >= T) return;

    const uint32_t r = t / tokens_per_rank;
    const uint32_t s = (r + num_ranks - local_rank) % num_ranks;     // ring step

    #pragma unroll 1
    for (uint32_t j = 0; j < K; ++j) {
        const int g = __ldg(routing_topk + t * K + j);              // global expert id
        // Same global->local map + filter as Phase 1 histogram (must stay in
        // lockstep, otherwise counts and scatter positions disagree).
        const uint32_t e = static_cast<uint32_t>(g - static_cast<int>(expert_base));
        if (g < 0 || e >= num_experts)
            continue;
        const uint32_t chunk = e * num_ranks + s;
        const int off = atomicAdd(&chunk_cursor[chunk], 1);
        const int pos = padded_starts[chunk] + off;
        gather_index[pos] = static_cast<int>(t);
        row_to_topk[pos] = static_cast<int>(topk_slot_offset + j);
    }
}

}  // namespace deep_gemm
