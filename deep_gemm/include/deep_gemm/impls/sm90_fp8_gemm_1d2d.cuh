#pragma once

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wunknown-attributes"

#include <cutlass/arch/barrier.h>
#include <cutlass/arch/reg_reconfig.h>

#include <cute/arch/cluster_sm90.hpp>
#include <cute/arch/copy_sm90_desc.hpp>
#include <cute/arch/copy_sm90_tma.hpp>

#include <deep_gemm/common/math.cuh>
#include <deep_gemm/common/utils.cuh>
#include <deep_gemm/common/tma_copy.cuh>
#include <deep_gemm/common/types.cuh>
#include <deep_gemm/mma/sm90.cuh>
#include <deep_gemm/epilogue/transform.cuh>
#include <deep_gemm/ptx/ld_st.cuh>
#include <deep_gemm/ptx/tma.cuh>
#include <deep_gemm/ptx/utils.cuh>
#include <deep_gemm/ptx/wgmma.cuh>
#include <deep_gemm/scheduler/gemm.cuh>
template <typename T, typename U>
__device__ __forceinline__ void cp_async4(T* smem_ptr, const U* glob_ptr) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{ cp.async.cg.shared.global [%0], [%1], %2; }\n"
        :: "r"(smem), "l"((const void*)glob_ptr), "n"(BYTES));
#else
    *smem_ptr = *reinterpret_cast<const T*>(glob_ptr);
#endif
}

// Zero-fill variant of cp_async4: copies `src_size` bytes from global to smem and
// zero-fills the rest of the 16-byte chunk in shared memory. When `src_size == 0`
// the entire 16-byte smem region is zero-filled by hardware (no global read is
// performed); the source pointer must still be a valid CUDA pointer (the HW
// validates the address but does not dereference). This matches PTX semantics
// for `cp.async.cg.shared.global ..., cp_size, src_size;` with `src_size <= cp_size`.
//
// Used for "pad rows" in the gather + per-rank-flag overlap path: rows whose
// `gather_index[i] < 0` (or `logical_m >= shape_m`) translate to `src_size = 0`
// here, so the corresponding smem A tile slice becomes all zeros and the math WG
// naturally produces `A · B = 0` for those output rows.
template <typename T, typename U>
__device__ __forceinline__ void cp_async4_zfill(T* smem_ptr, const U* glob_ptr, uint32_t src_size) {
#if __CUDACC_VER_MAJOR__ >= 11 && __CUDA_ARCH__ >= 800
    const int BYTES = 16;
    uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    asm volatile(
        "{ cp.async.cg.shared.global [%0], [%1], %2, %3; }\n"
        :: "r"(smem), "l"((const void*)glob_ptr), "n"(BYTES), "r"(src_size));
#else
    if (src_size > 0)
        *smem_ptr = *reinterpret_cast<const T*>(glob_ptr);
    else
        *smem_ptr = T{};
#endif
}
namespace deep_gemm {

template <uint32_t kNumFormerIters, uint32_t kGap, uint32_t kEnd, typename func_t>
CUTLASS_DEVICE void dispatch_num_former_iters(uint32_t num_former_iters, const func_t& func) {
    if (num_former_iters == kNumFormerIters) {
        func(cute::Int<kNumFormerIters>{});
        return;
    }

    if constexpr (kNumFormerIters + kGap <= kEnd)
        dispatch_num_former_iters<kNumFormerIters + kGap, kGap, kEnd>(num_former_iters, func);
}

// ---------------------------------------------------------------------------
// Barrier debug helpers — throttled edition
//
// Controls (all have compile-time defaults, override via -D flag if needed):
//   DG_BARRIER_DEBUG    : 0=off, 1=on  (default 1)
//   DG_DBG_BLOCK_ID     : only this blockIdx.x prints  (default 0)
//   DG_DBG_MAX_STAGES   : only stage < this value prints  (default 2)
//   DG_DBG_PRINT_TID    : which threadIdx.x inside each WG is the "reporter"
//                         Producer WG reporter = kNumMathThreads + DG_DBG_PRINT_TID  (default 0)
//                         Math WG reporter     = DG_DBG_PRINT_TID
//
// Net effect: at most (DG_DBG_MAX_STAGES * <num_print_sites>) lines per run —
// easily fits in the 1 MB CUDA printf buffer.
// ---------------------------------------------------------------------------
#ifndef DG_BARRIER_DEBUG
#define DG_BARRIER_DEBUG 0
#endif
#ifndef DG_DBG_BLOCK_ID
#define DG_DBG_BLOCK_ID 0
#endif
#ifndef DG_DBG_MAX_STAGES
#define DG_DBG_MAX_STAGES 4u
#endif
#ifndef DG_DBG_PRINT_TID
#define DG_DBG_PRINT_TID 0
#endif

#if DG_BARRIER_DEBUG
// Print barrier pointer info.
// DG_BARRIER_DEBUG=1: only print generic pointer address (safe, no shared mem access).
// DG_BARRIER_DEBUG=2: also read and decode the mbarrier word (requires valid smem layout).
__device__ __forceinline__ void dg_print_barrier(
        const char* tag,
        uint32_t blockX, uint32_t tidX,
        uint32_t stage,
        const cutlass::arch::ClusterTransactionBarrier* ptr,
        uint32_t wait_phase) {
    // Use generic address (uintptr_t) to avoid any cvta instruction that could fault.
    const uintptr_t generic_addr = reinterpret_cast<uintptr_t>(ptr);
#if DG_BARRIER_DEBUG >= 2
    // PTX mbarrier layout (SM90):
    //   bit 0     : phase/parity bit (current phase)
    //   bits 1-11 : pending arrive count  (how many more arrives needed)
    //   bits 12-31: pending tx bytes      (how many more bytes the HW must deliver)
    uint32_t smem_addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t raw = 0;
    asm volatile("ld.shared.b64 %0, [%1];" : "=l"(raw) : "r"(smem_addr));
    uint32_t parity       = (uint32_t)(raw & 1u);
    uint32_t arrive_rem   = (uint32_t)((raw >> 1) & 0x7FFu);
    uint32_t tx_rem       = (uint32_t)((raw >> 12) & 0xFFFFFu);
    printf("[DBG-BARRIER] blk=%u tid=%u  %-22s  stage=%u  ptr=0x%llx"
           "  raw=0x%012llx  parity=%u(wait=%u)  arrive_rem=%u  tx_rem=%u\n",
           blockX, tidX, tag, stage, (unsigned long long)generic_addr,
           (unsigned long long)raw,
           parity, wait_phase, arrive_rem, tx_rem);
#else
    printf("[DBG-BARRIER] blk=%u tid=%u  %-22s  stage=%u  ptr=0x%llx  wait_phase=%u\n",
           blockX, tidX, tag, stage, (unsigned long long)generic_addr, wait_phase);
#endif
}
// Guard: only block DG_DBG_BLOCK_ID, only stage < DG_DBG_MAX_STAGES,
//        only thread threadIdx.x == reporter_tid (passed by caller).
// reporter_tid is different per WG so the macro takes it explicitly.
#define DG_DBG_BARRIER(tag, stage, ptr, wphase, reporter_tid)          \
    do {                                                                \
        if (blockIdx.x == (uint32_t)DG_DBG_BLOCK_ID &&                 \
            (stage) < DG_DBG_MAX_STAGES &&                             \
            threadIdx.x == (uint32_t)(reporter_tid)) {                 \
            dg_print_barrier((tag), blockIdx.x, threadIdx.x,           \
                             (stage), (ptr), (wphase));                \
        }                                                               \
    } while (0)
#else
#define DG_DBG_BARRIER(tag, stage, ptr, wphase, reporter_tid) ((void)0)
#endif

template <cute::UMMA::Major kMajorSFB,
          uint32_t SHAPE_M, uint32_t SHAPE_N, uint32_t SHAPE_K,
          uint32_t kNumGroups,
          uint32_t BLOCK_M, uint32_t BLOCK_N, uint32_t BLOCK_K,
          uint32_t kSwizzleAMode, uint32_t kSwizzleBMode, uint32_t kSwizzleDMode,
          uint32_t kNumStages,
          uint32_t kNumTMAThreads, uint32_t kNumMathThreads,
          uint32_t kNumTMAMulticast, bool kIsTMAMulticastOnA,
          uint32_t kNumSMs, GemmType kGemmType,
          bool kSFAIsMNMajor, bool kHasGatherIndex, bool kHasRankFlags,
          bool kCombineScatter,
          typename epilogue_type_t>
CUTLASS_GLOBAL __launch_bounds__(kNumTMAThreads + kNumMathThreads, 1) void
sm90_fp8_gemm_1d2d_impl(float* sfb, int* grouped_layout,
                        uint32_t shape_m, uint32_t shape_n, uint32_t shape_k,
                        const __nv_fp8_e4m3* __restrict__ gmem_a,
                        uint32_t stride_a,          // row stride of A in fp8 elements
                        const float* __restrict__ gmem_sfa,
                        uint32_t stride_sfa,        // MN-major: K-scale stride; raw row-major: row stride
                        const int* __restrict__ gather_index,
                        // Per-rank ready-flag overlap (optional). All four must be set together
                        // (or all unset). See docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md.
                        //   rank_flags      : (num_ranks,) int64 on global memory; written to
                        //                     `rank_flag_epoch` by the comm stream once that rank's
                        //                     tokens are visible in `gmem_a`.
                        //   tile_rank       : (ceil(shape_m/BLOCK_M),) int32. Non-negative entries
                        //                     are the unique rank that the corresponding M tile
                        //                     depends on. Negative entries mean mixed/unknown ranks
                        //                     and conservatively wait for all ranks.
                        //   num_ranks       : runtime rank count, must satisfy `num_ranks <= 8`.
                        //   rank_flag_epoch : monotonically increasing epoch; the kernel spins
                        //                     until `ld.acquire.sys(rank_flags[r]) >= epoch`.
                        //                     Eliminates the need to zero rank_flags each round.
                        const uint64_t* __restrict__ rank_flags,
                        const int* __restrict__ tile_rank,
                        uint32_t num_ranks,
                        uint64_t rank_flag_epoch,
                        const int* __restrict__ combine_row_topk,
                        const float* __restrict__ combine_topk_scores,
                        const uint64_t* __restrict__ combine_buffer_ptrs,
                        uint32_t combine_tokens_per_rank,
                        uint32_t combine_top_k,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_a,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_b,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_d,
                        const __grid_constant__ cute::TmaDescriptor tensor_map_sfa) {
#if (defined(__CUDA_ARCH__) and (__CUDA_ARCH__ >= 900)) or defined(__CLION_IDE__)
    static_assert(not kHasRankFlags or kHasGatherIndex, "rank flags require gather_index");
    static_assert(not kCombineScatter or kHasGatherIndex, "combine-scatter requires gather_index");

    // Scaling checks
    DG_STATIC_ASSERT(BLOCK_K == 128, "Only support per-128-channel FP8 scaling");
    DG_STATIC_ASSERT(
        math::constexpr_ceil_div(BLOCK_N, BLOCK_K) == 1 or
        (math::constexpr_gcd(BLOCK_N, BLOCK_K) == BLOCK_N - BLOCK_K), "Too much B scales in a single block");

    // Types
    using WGMMA = typename mma::sm90::FP8MMASelector<BLOCK_N>::type;
    using Barrier = cutlass::arch::ClusterTransactionBarrier;
    DG_STATIC_ASSERT(BLOCK_M % WGMMA::M == 0 or BLOCK_M < WGMMA::M, "Invalid block size");

    // Overwrite shape constants if the compiler gives
    shape_m = SHAPE_M != 0 ? SHAPE_M : shape_m;
    shape_n = SHAPE_N != 0 ? SHAPE_N : shape_n;
    shape_k = SHAPE_K != 0 ? SHAPE_K : shape_k;

    // Shared memory
    static constexpr bool kMustUseUniformedScaleB = (BLOCK_K % BLOCK_N == 0);
    static constexpr uint32_t SMEM_D_SIZE = math::constexpr_align(BLOCK_M * BLOCK_N * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024u);
    static constexpr uint32_t SMEM_A_SIZE_PER_STAGE = BLOCK_M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_B_SIZE_PER_STAGE = BLOCK_N * BLOCK_K * sizeof(__nv_fp8_e4m3);
    static constexpr uint32_t SMEM_SFA_SIZE_PER_STAGE = BLOCK_M * sizeof(float);
    static constexpr uint32_t ALIGNED_SMEM_SFA_SIZE_PER_STAGE = math::constexpr_align(SMEM_SFA_SIZE_PER_STAGE, 128u);
    const uint32_t shape_k_scales = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t shape_n_sfb = math::ceil_div(shape_n, BLOCK_K);
    const uint32_t smem_sfb_size = math::align<uint32_t>(shape_k_scales * (kMustUseUniformedScaleB ? 1 : 2) * sizeof(float), sizeof(Barrier));

    // NOTES: Make sure we have enough shared memory for WGMMA padding
    static constexpr uint32_t WGMMA_A_SIZE_PER_STAGE = WGMMA::M * BLOCK_K * sizeof(__nv_fp8_e4m3);
    DG_STATIC_ASSERT(WGMMA_A_SIZE_PER_STAGE <= SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE * kNumStages, "Memory Out of bound for WGMMA");

    // Configs
    const uint32_t num_total_k_blocks = math::ceil_div(shape_k, BLOCK_K);
    const uint32_t warp_idx = __shfl_sync(0xffffffff, threadIdx.x / 32, 0);
    const uint32_t lane_idx = ptx::get_lane_idx();

    // Prefetch TMA descriptors at the very beginning
    if (warp_idx == kNumMathThreads / 32 and cute::elect_one_sync()) {
        cute::prefetch_tma_descriptor(&tensor_map_a);
        cute::prefetch_tma_descriptor(&tensor_map_b);
        cute::prefetch_tma_descriptor(&tensor_map_sfa);
        cute::prefetch_tma_descriptor(&tensor_map_d);
    }
    __syncwarp();

    // Align to 1024 bytes for swizzle-128B
    extern __shared__ __align__(1024) uint8_t smem_buffer[];
    DG_STATIC_ASSERT(SMEM_D_SIZE % 1024 == 0, "Shared memory of A/B must be aligned to 1024 bytes");

    // Data on shared memory
    auto smem_d = reinterpret_cast<__nv_bfloat16*>(smem_buffer);
    auto smem_a = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + i * SMEM_A_SIZE_PER_STAGE);
    });
    auto smem_b = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<__nv_fp8_e4m3*>(smem_buffer + SMEM_D_SIZE + kNumStages * SMEM_A_SIZE_PER_STAGE + i * SMEM_B_SIZE_PER_STAGE);
    });
    constexpr uint32_t SMEM_SF_OFFSET = SMEM_D_SIZE + kNumStages * (SMEM_A_SIZE_PER_STAGE + SMEM_B_SIZE_PER_STAGE);
    auto smem_sfa = utils::PatternVisitor([&](const uint32_t& i) {
        return reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + i * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);
    });
    auto smem_sfb = reinterpret_cast<float*>(smem_buffer + SMEM_SF_OFFSET + kNumStages * ALIGNED_SMEM_SFA_SIZE_PER_STAGE);

    // Fill barriers
    // Layout: [full_barriers_b(0..kNumStages-1)] [full_barriers_a(kNumStages..2*kNumStages-1)] [empty_barriers(2*kNumStages..3*kNumStages-1)]
    // Single producer WG (128 threads) now owns both TMA and cp.async:
    //   full_barriers_b: produced by one elected thread of a rotating warp via TMA (B only)
    //   full_barriers_a: produced by all 128 threads via cp.async.mbarrier.arrive.noinc (A + sfa)
    //   empty_barriers:  signaled by Math WG after consuming
    auto barrier_start_ptr = reinterpret_cast<Barrier*>(reinterpret_cast<uint8_t*>(smem_sfb) + smem_sfb_size);
    // auto full_barriers_b   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    // auto full_barriers_a   = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + kNumStages + i; });
    auto full_barriers = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + i; });
    auto empty_barriers    = utils::PatternVisitor([&](const uint32_t& i) { return barrier_start_ptr + 1 * kNumStages + i; });

    // Per-rank "ready seen" cache for the gather + per-rank-flag overlap path.
    // 8 entries (kNumRanksMax = 8 = NVL8 upper bound). All 128 producer-WG
    // threads check s_rank_seen[r] via volatile smem read; if 0 they each
    // spin on rank_flags[r] with ld.acquire.sys. The elected thread writes
    // s_rank_seen[r] = 1 after the spin so subsequent tiles skip via cache hit.
    //
    // Lives in dynamic smem after the barriers; the host's smem_size reservation
    // for `kNumMaxStages * 8 * 3` barriers (see `get_pipeline_config` in
    // sm90.hpp) leaves >256 B slack vs. the actual `2 * kNumStages` barriers we
    // use, so 32 B for `s_rank_seen` fits without bumping `smem_size`.
    static constexpr uint32_t kNumRanksMax = 8;
    auto smem_tail = reinterpret_cast<uint8_t*>(barrier_start_ptr + 2 * kNumStages);
    uint32_t* s_rank_seen = nullptr;
    if constexpr (kHasRankFlags)
        s_rank_seen = reinterpret_cast<uint32_t*>(smem_tail);

#if DG_BARRIER_DEBUG
    // Diagnose SMEM layout: print offsets and total usage (block 0, thread 0 only).
    if (blockIdx.x == 0 && threadIdx.x == 0) {
        uint32_t smem_base      = static_cast<uint32_t>(__cvta_generic_to_shared(smem_buffer));
        uint32_t barrier_off    = static_cast<uint32_t>(
            reinterpret_cast<uint8_t*>(barrier_start_ptr) - smem_buffer);
        uint32_t barrier_end_off= barrier_off + 3u * kNumStages * 8u;
        printf("[DBG-SMEM-LAYOUT] blk=0 tid=0  smem_base=0x%x"
               "  SMEM_D=%u A_per=%u B_per=%u SFA_per=%u"
               "  sfb_size=%u  barrier_off=%u barrier_end=%u"
               "  kNumStages=%u\n",
               smem_base,
               (unsigned)SMEM_D_SIZE,
               (unsigned)SMEM_A_SIZE_PER_STAGE,
               (unsigned)SMEM_B_SIZE_PER_STAGE,
               (unsigned)ALIGNED_SMEM_SFA_SIZE_PER_STAGE,
               (unsigned)smem_sfb_size,
               barrier_off, barrier_end_off,
               (unsigned)kNumStages);
    }
#endif

    // Initialize barriers
    DG_STATIC_ASSERT(kNumTMAMulticast <= 32, "Too many TMA multicast");
    if (warp_idx == kNumMathThreads / 32 + 1 and cute::elect_one_sync()) {
        // NOTES: we always use `lane_idx` to arrive for the `lane_idx`-th CTA in the cluster,
        // even with TMA multicast disabled, we want to make the behavior aligned
        #pragma unroll
        for (uint32_t i = 0; i < kNumStages; ++ i) {
            // full_barriers_b[i]->init(1);    // one elected thread (rotating warp) issues TMA for B
            // full_barriers_a[i]->init(128);  // all 128 producer-WG threads arrive via cp.async.mbarrier.arrive.noinc
            full_barriers[i]->init(129);
            empty_barriers[i]->init(kNumTMAMulticast * kNumMathThreads / 32);
        }

        if constexpr (kHasRankFlags) {
            // Zero-init the per-rank "seen" cache for rank-flag kernels only.
            #pragma unroll
            for (uint32_t i = 0; i < kNumRanksMax; ++ i)
                s_rank_seen[i] = 0;
        }

        // Make initialized barrier visible in async proxy
        cutlass::arch::fence_barrier_init();
    }

    // Synchronize all threads to make barrier visible in normal memory model
    (kNumTMAMulticast > 1) ? cute::cluster_sync() : __syncthreads();

    // Register reconfigurations
    // Keep enough producer registers for fused gather address arithmetic while
    // giving the math warp-groups a spill-free budget.
    constexpr uint32_t kNumProducerRegisters = 48;
    constexpr uint32_t kNumMathRegisters     = kNumMathThreads == 128 ? 248 : 224;

    static constexpr uint32_t kCpAsyncWidth   = 16;
    static constexpr uint32_t kCpAsyncThreads = 128;
    static constexpr uint32_t kAItersPerThread = SMEM_A_SIZE_PER_STAGE / kCpAsyncThreads / kCpAsyncWidth;
    static constexpr uint32_t kSFAItersPerThread = math::constexpr_ceil_div(BLOCK_M, kCpAsyncThreads);
    static_assert(kAItersPerThread >= 1,
                  "Not enough cp.async threads for A load");
    static_assert(SMEM_A_SIZE_PER_STAGE % (kCpAsyncThreads * kCpAsyncWidth) == 0,
                  "A smem size must be divisible by (128 * 16)");
    static constexpr uint32_t kAThreadsPerRow = BLOCK_K / kCpAsyncWidth;
    static constexpr uint32_t kRowsPerAIter = kCpAsyncThreads / kAThreadsPerRow;

    // Wait for primary kernel completion
    cudaGridDependencySynchronize();

    // Block scheduler
    uint32_t m_block_idx, n_block_idx;
    auto scheduler = sched::Scheduler<kGemmType, BLOCK_M, BLOCK_N, kNumGroups, kNumTMAMulticast, kIsTMAMulticastOnA, kNumSMs>(shape_m, shape_n, shape_k, grouped_layout);

    // Pipeline phase tracker
    uint32_t stage_idx = 0, phase = 0;
    auto advance_pipeline = [&](uint32_t& k_block_idx) {
        ++ k_block_idx;
        stage_idx = stage_idx == kNumStages - 1 ? 0 : stage_idx + 1;
        phase ^= stage_idx == 0;
    };

    // -------------------------------------------------------------------------
    // Producer WG (merged TMA + cp.async, 128 threads, 4 warps):
    //   threadIdx.x in [kNumMathThreads, kNumMathThreads + 127]
    //   warp_idx    in [kNumMathThreads/32, kNumMathThreads/32 + 3]
    //
    // Per k-tile responsibilities:
    //   (1) One elected thread of a rotating warp issues TMA for B.
    //       The TMA-issuing warp is `warp_in_wg = k_block_idx % 4`, so warp 0 issues
    //       the first k-tile's TMA, warp 1 the second, warp 2 the third, warp 3 the
    //       fourth, and then wraps around. This spreads TMA-descriptor work across warps
    //       and keeps each warp's register pressure uniform.
    //   (2) All 128 threads issue cp.async for A (with swizzle-128B layout) and sfa.
    //   (3) All 128 threads call cp.async.mbarrier.arrive.noinc on full_barriers_a.
    //       The hardware decrements the arrive count for each thread once that thread's
    //       prior cp.async ops complete, so the producer never has to block on
    //       cp.async.wait_group — it proceeds straight to the next stage's empty-wait.
    // -------------------------------------------------------------------------
    if (warp_idx >= kNumMathThreads / 32 and warp_idx < kNumMathThreads / 32 + 4) {
        cutlass::arch::warpgroup_reg_dealloc<kNumProducerRegisters>();

        const uint32_t tid_in_wg   = threadIdx.x - kNumMathThreads;            // [0, 128)
        const uint32_t warp_in_wg  = warp_idx - kNumMathThreads / 32;          // [0, 4)

        // `kPadSentinel == (uint32_t)-1` doubles as the natural cast result of
        // `gather_index[i] = -1`, the "pad row" signal documented in
        // sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md. We also force the
        // sentinel for `logical_m >= shape_m` so the trailing partial tile
        // can't issue out-of-bounds reads.
        constexpr uint32_t kPadSentinel = 0xFFFFFFFFu;

        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            constexpr bool kWithGroupOffsetA = kGemmType == GemmType::MGroupedMasked;
            const uint32_t m_global_base = scheduler.get_global_idx<kWithGroupOffsetA>(shape_m, BLOCK_M, m_block_idx);

            // ----- Per-rank ready-flag wait (gather + overlap path only) -----
            //
            // All 128 producer-WG threads read tile_rank (L2 broadcast). For
            // unique-rank tiles they check s_rank_seen[r] via volatile smem read.
            // expert_srank_padding=false can produce mixed-rank tiles; those use a
            // negative tile_rank sentinel and wait for every rank.
            //
            // Fast path (cached rank): volatile read sees 1 → skip.
            //   Cost: __ldg + volatile smem read + branch ≈ 0.1 us.
            //
            // Slow path (new rank): all 128 threads each spin on rank_flags[r]
            //   with ld.acquire.sys. Each thread independently gets system-scope
            //   visibility — subsequent cp.async from gmem_a sees remote writes.
            //   No NamedBarrier or membar.sys needed.
            //   Cost: 128 threads × ld.acquire.sys broadcast ≈ 1 us.
            //
            // See docs/sm90_fp8_gemm_1d2d_gather_index_rank_overlap.md §14.
            if constexpr (kHasRankFlags) {
                auto wait_rank_ready = [&](const uint32_t r) {
                    DG_TRAP_ONLY_DEVICE_ASSERT(r < num_ranks and r < kNumRanksMax);
                    if (*reinterpret_cast<volatile uint32_t*>(s_rank_seen + r) == 0) {
                        while (deep_gemm::ptx::ld_acq_sys(rank_flags + r) < rank_flag_epoch) {
                        }
                        if (warp_in_wg == 0 and cute::elect_one_sync()) {
                            s_rank_seen[r] = 1;
                        }
                    }
                };

                const int tile_r = __ldg(tile_rank + m_block_idx);
                if (tile_r >= 0) {
                    wait_rank_ready(static_cast<uint32_t>(tile_r));
                } else {
                    #pragma unroll
                    for (uint32_t r = 0; r < kNumRanksMax; ++ r) {
                        if (r < num_ranks)
                            wait_rank_ready(r);
                    }
                }
            }
            // -----------------------------------------------------------------

            uint32_t source_m_for_a[kHasGatherIndex ? kAItersPerThread : 1];
            uint32_t source_m_for_sfa[kHasGatherIndex ? kSFAItersPerThread : 1];
            if constexpr (kHasGatherIndex) {
                #pragma unroll
                for (uint32_t i = 0; i < kAItersPerThread; ++i) {
                    const uint32_t row = i * kRowsPerAIter + tid_in_wg / kAThreadsPerRow;
                    const uint32_t logical_m = m_global_base + row;
                    if (logical_m >= shape_m) {
                        source_m_for_a[i] = kPadSentinel;
                    } else {
                        const int g = __ldg(gather_index + logical_m);
                        source_m_for_a[i] = (g < 0) ? kPadSentinel : static_cast<uint32_t>(g);
                    }
                }
                #pragma unroll
                for (uint32_t i = 0; i < kSFAItersPerThread; ++i) {
                    const uint32_t row = i * kCpAsyncThreads + tid_in_wg;
                    const uint32_t logical_m = m_global_base + row;
                    if (row >= BLOCK_M or logical_m >= shape_m) {
                        source_m_for_sfa[i] = kPadSentinel;
                    } else {
                        const int g = __ldg(gather_index + logical_m);
                        source_m_for_sfa[i] = (g < 0) ? kPadSentinel : static_cast<uint32_t>(g);
                    }
                }
            }

            for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                const bool is_tma_multicast_valid = scheduler.is_tma_multicast_valid(m_block_idx);
                const uint32_t num_tma_multicast_a = (kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                const uint32_t num_tma_multicast_b = (not kIsTMAMulticastOnA and is_tma_multicast_valid) ? kNumTMAMulticast : 1;
                DG_STATIC_ASSERT(kNumTMAMulticast <= 2, "Scheduler does not support > 2 TMA multicast");

                // Wait for Math WG to finish consuming this stage
                empty_barriers[stage_idx]->wait(phase ^ 1);

                constexpr bool kIsBatchedMM = (kGemmType == GemmType::Batched);
                const uint32_t batch_idx    = (kIsBatchedMM ? scheduler.current_group_idx : 0);
                const uint32_t k_idx        = k_block_idx * BLOCK_K;
                // auto& full_b = *full_barriers_b[stage_idx];
                auto &full=*full_barriers[stage_idx];

                // (1) Rotating-warp TMA: only the warp whose index within the WG matches
                //     (k_block_idx % 4) issues TMA for B. Within that warp, elect_one_sync
                //     picks a single thread. full_barriers_b has init count = 1, matching
                //     the single arrive_and_expect_tx below.
                const uint32_t tma_warp = k_block_idx & 3u;  // == k_block_idx % 4
                if (warp_in_wg == tma_warp and cute::elect_one_sync()) {
                    tma::copy<BLOCK_K, BLOCK_N, kSwizzleBMode, __nv_fp8_e4m3, kIsBatchedMM>(&tensor_map_b, &full,
                             smem_b[stage_idx],
                             k_idx,
                             scheduler.get_global_idx<true>(shape_n, BLOCK_N, n_block_idx, m_block_idx),
                             num_tma_multicast_b, batch_idx);
                    full.arrive_and_expect_tx(SMEM_B_SIZE_PER_STAGE);
                    // NOTE: No printf here — producer WG only has 40 registers (warpgroup_reg_dealloc),
                    //       not enough for printf/vfprintf_internal which needs ~30-50 registers.
                }

                // (2a) All 128 threads load A via cp.async with swizzle-128B layout.
                //      Each 16B cp.async covers one contiguous chunk; swizzle col XOR (row%8)*16
                //      to match the WGMMA layout_type=1 (B128 swizzle) descriptor.
                //
                // Iter-major (row-group) mapping:
                //      linear = (i * 128 + tid_in_wg) * 16
                //   => 8 consecutive lanes in a warp cover one full 128B row (all 8 cols),
                //      and each warp covers 4 consecutive rows per iter. The HW coalesces
                //      the 32-lane requests into 4 x 128B cache-line fetches per warp-iter
                //      (fully-utilized transactions), vs. the naive tid-major mapping which
                //      scatters across 16 cache lines per warp-iter and relies on L2 reuse
                //      across iters.
                //
                //   Row covered per iter (across all 128 threads): kRowsPerIter = 128 / 8 = 16.
                //   Over kAItersPerThread = BLOCK_M / 16 iters, rows [0, BLOCK_M) are covered
                //   exactly once — verified for BLOCK_M ∈ {16, 32, 64, 128, 256}.
                __nv_fp8_e4m3* dst_a = smem_a[stage_idx];
                if constexpr (kHasGatherIndex) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAItersPerThread; ++i) {
                        const uint32_t row = i * kRowsPerAIter + tid_in_wg / kAThreadsPerRow;
                        const uint32_t col = (tid_in_wg % kAThreadsPerRow) * kCpAsyncWidth;
                        const uint32_t source_m = source_m_for_a[i];
                        // Pad-aware cp.async: when `source_m == kPadSentinel` the row is
                        // either out-of-shape_m or explicitly tagged by the host as pad. Pass
                        // `src_size = 0` to make HW zero-fill the 16 B smem slot without
                        // dereferencing `src`.
                        const bool is_pad = (source_m == kPadSentinel);
                        const uint32_t src_size = is_pad ? 0u : kCpAsyncWidth;
                        const __nv_fp8_e4m3* src_a = is_pad
                            ? gmem_a
                            : gmem_a + source_m * stride_a + k_idx + col;
                        cp_async4_zfill(dst_a + row * BLOCK_K + (col ^ ((row % 8) * 16)),
                                        src_a, src_size);
                    }
                } else {
                    #pragma unroll
                    for (uint32_t i = 0; i < kAItersPerThread; ++i) {
                        const uint32_t row = i * kRowsPerAIter + tid_in_wg / kAThreadsPerRow;
                        const uint32_t col = (tid_in_wg % kAThreadsPerRow) * kCpAsyncWidth;
                        cp_async4(dst_a + row * BLOCK_K + (col ^ ((row % 8) * 16)),
                                  gmem_a + (m_global_base + row) * stride_a + k_idx + col);
                    }
                }

                // (2b) Threads cooperatively load sfa rows; each thread may cover multiple
                //      rows when BLOCK_M > 128.
                //
                //      Pad-aware: rows with `source_m_for_sfa[i] == kPadSentinel` (either
                //      out-of-shape_m or explicit `gather_index < 0`) issue a 4 B cp.async
                //      with `src_size = 0`, letting HW zero-fill the smem slot (so that
                //      `scale_a == 0` for that row → `A · B · scale = 0` in the math WG).
                //      Fully past-tile rows (`row >= BLOCK_M`) skip the cp.async entirely.
                //
                //      Out-of-bound rows that are skipped still participate in
                //      `cp.async.mbarrier.arrive.noinc` correctly: a thread with no
                //      outstanding cp.async in this group resolves its arrive immediately.
                if constexpr (kHasGatherIndex) {
                    #pragma unroll
                    for (uint32_t i = 0; i < kSFAItersPerThread; ++i) {
                        const uint32_t row = i * kCpAsyncThreads + tid_in_wg;
                        if (row < BLOCK_M) {
                            const uint32_t sfa_k_idx = scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::SF_K>(shape_k_scales, 1, k_block_idx);
                            const uint32_t source_m = source_m_for_sfa[i];
                            const bool is_pad_sfa = (source_m == kPadSentinel);
                            const uint32_t src_size_sfa = is_pad_sfa ? 0u : 4u;
                            float* dst_sfa = smem_sfa[stage_idx] + row;
                            // For pad rows, fall back to `gmem_sfa` as a known-valid source
                            // pointer; HW won't dereference it because src_size = 0.
                            const float* src_sfa = is_pad_sfa
                                ? gmem_sfa
                                : (kSFAIsMNMajor
                                    ? gmem_sfa + sfa_k_idx * stride_sfa + source_m
                                    : gmem_sfa + source_m * stride_sfa + sfa_k_idx);
                            asm volatile(
                                "cp.async.ca.shared.global [%0], [%1], 4, %2;\n"
                                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_sfa))),
                                   "l"(reinterpret_cast<const void*>(src_sfa)),
                                   "r"(src_size_sfa));
                        }
                    }
                } else {
                    #pragma unroll
                    for (uint32_t i = 0; i < kSFAItersPerThread; ++i) {
                        const uint32_t row = i * kCpAsyncThreads + tid_in_wg;
                        if (row < BLOCK_M and m_global_base + row < shape_m) {
                            const uint32_t sfa_k_idx = scheduler.template get_global_idx<kWithGroupOffsetA, sched::IndexType::SF_K>(shape_k_scales, 1, k_block_idx);
                            const uint32_t source_m = m_global_base + row;
                            float* dst_sfa = smem_sfa[stage_idx] + row;
                            const float* src_sfa = kSFAIsMNMajor
                                ? gmem_sfa + sfa_k_idx * stride_sfa + source_m
                                : gmem_sfa + source_m * stride_sfa + sfa_k_idx;
                            asm volatile(
                                "cp.async.ca.shared.global [%0], [%1], %2;\n"
                                :: "r"(static_cast<uint32_t>(__cvta_generic_to_shared(dst_sfa))),
                                   "l"(reinterpret_cast<const void*>(src_sfa)),
                                   "n"(4));
                        }
                    }
                }

                // (3) Commit this stage's cp.async group and asynchronously arrive on
                //     full_barriers_a. All 128 threads call cp.async.mbarrier.arrive.noinc
                //     (barrier init count = 128); each thread's arrive is deferred by HW
                //     until that thread's prior cp.async ops complete. The producer never
                //     blocks on cp.async.wait_group — it proceeds straight to the next
                //     stage's empty-wait (real async pipelining).
                asm volatile("cp.async.commit_group;\n" ::: "memory");
                ptx::cp_async_mbarrier_arrive_noinc(full_barriers[stage_idx]);
            }
        }

        // To safely deconstruct distributed shared barriers, we need another round of empty waits
        if constexpr (kNumTMAMulticast > 1) {
            for (uint32_t i = 0; i < kNumStages; advance_pipeline(i))
                empty_barriers[stage_idx]->wait(phase ^ 1);
        }

    } else {
        // Math warp-groups for WGMMA
        cutlass::arch::warpgroup_reg_alloc<kNumMathRegisters>();

        // NOTES: use `__shfl_sync` to encourage NVCC to use unified registers
        const auto math_wg_idx = __shfl_sync(0xffffffff, threadIdx.x / 128, 0);
        const auto r_0 = warp_idx * 16 + lane_idx / 4, r_1 = r_0 + 8;

        auto a_desc = mma::sm90::make_smem_desc(smem_a[0] + math_wg_idx * WGMMA::M * BLOCK_K, 1); 
        auto b_desc = mma::sm90::make_smem_desc(smem_b[0], 1);
        const uint32_t a_desc_lo = __shfl_sync(0xffffffff, a_desc.reg32_[0], 0);
        const uint32_t b_desc_lo = __shfl_sync(0xffffffff, b_desc.reg32_[0], 0);

        // Persistently schedule over blocks
        while (scheduler.get_next_block(m_block_idx, n_block_idx)) {
            // Decide the number of scales B to load
            DG_TRAP_ONLY_DEVICE_ASSERT(shape_n % 8 == 0);
            uint32_t num_former_iters = BLOCK_N / 8, num_full_iters = num_former_iters;
            if constexpr (not kMustUseUniformedScaleB) {
                num_former_iters = min(BLOCK_N, BLOCK_K - n_block_idx * BLOCK_N % BLOCK_K) / 8;
                num_full_iters = min(shape_n - n_block_idx * BLOCK_N, BLOCK_N) / 8;
            }
            uint32_t num_sfb = shape_k_scales * (num_former_iters >= num_full_iters ? 1 : 2);

            // Load B scales with math warp-groups
            // NOTES: except the first warp, we want to overlap loading B scales with TMA stores between tasks
            if (threadIdx.x >= 32) {
                auto previous_group_offset = scheduler.template get_global_idx<true, sched::IndexType::SF_K>(shape_n_sfb * shape_k_scales, 0, 0, m_block_idx);
                const uint32_t stride_n_sfb = kMajorSFB == cute::UMMA::Major::MN ? 1 : shape_k_scales;
                const uint32_t stride_k_sfb = kMajorSFB == cute::UMMA::Major::MN ? shape_n_sfb : 1;
                auto local_sfb = sfb + previous_group_offset + ((n_block_idx * BLOCK_N) / BLOCK_K) * stride_n_sfb;

                #pragma unroll
                for (uint32_t i = threadIdx.x - 32; i < num_sfb; i += kNumMathThreads - 32)
                    ptx::st_shared(smem_sfb + i, i < shape_k_scales ? local_sfb[i * stride_k_sfb] : local_sfb[(i - shape_k_scales) * stride_k_sfb + stride_n_sfb]);
            }
            cutlass::arch::NamedBarrier::sync(kNumMathThreads, 0);

            // Accumulation for WGMMA or CUDA promotion
            constexpr uint32_t WAVE_BLOCK_M = BLOCK_M <= WGMMA::M ? BLOCK_M : WGMMA::M * 2;
            DG_STATIC_ASSERT(BLOCK_M % WAVE_BLOCK_M == 0, "Invalid block sizes");
            float accum[WGMMA::kNumAccum], final_accum[WGMMA::kNumAccum * (BLOCK_M / WAVE_BLOCK_M)] = {0};
            
            // Pick threads whose WGMMA results are to be stored in shared memory
            DG_STATIC_ASSERT(BLOCK_M >= 64 or kNumMathThreads == 128, "Only one math warp group for `BLOCK_M < 64`");
            constexpr uint32_t kNumWGMMAStoreThreads = WAVE_BLOCK_M * (128 / WGMMA::M);
            const bool do_wgmma_store = BLOCK_M >= WGMMA::M or warp_idx < kNumWGMMAStoreThreads / 32;

            // Empty barrier arrival
            auto empty_barrier_arrive = [&]() {
                if constexpr (kNumTMAMulticast == 1) {
                    lane_idx == 0 ? empty_barriers[stage_idx]->arrive() : void();
                } else {
                    auto target_cta = scheduler.is_peer_cta_alive ? lane_idx : cute::block_rank_in_cluster();
                    lane_idx < kNumTMAMulticast ? empty_barriers[stage_idx]->arrive(target_cta) : void();
                }
            };

            // Skip useless computations
            if (scheduler.is_computation_valid(m_block_idx, math_wg_idx * WGMMA::M)) {
                // The compiler must know the dynamic variable `num_former_iters`'s real value
                constexpr bool kShouldOptimize = BLOCK_K / math::constexpr_gcd(BLOCK_K, BLOCK_N) <= 4 and not kMustUseUniformedScaleB;
                constexpr uint32_t kGap = math::constexpr_gcd(BLOCK_K, BLOCK_N) / 8;
                constexpr uint32_t kEnd = kShouldOptimize ? BLOCK_K / 8 : 0;

                // Dispatch `num_former_iters` and launch MMAs
                dispatch_num_former_iters<0, kGap, kEnd>(kShouldOptimize ? num_former_iters : 0, [&](auto _) {
                    #pragma unroll 8
                    for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                        const auto a_desc_base_lo = a_desc_lo + stage_idx * (SMEM_A_SIZE_PER_STAGE / 16);
                        const auto b_desc_base_lo = b_desc_lo + stage_idx * (SMEM_B_SIZE_PER_STAGE / 16);

                        // Read B scales
                        float scale_b_0 = ptx::ld_shared(smem_sfb + k_block_idx), scale_b_1;
                        // NOTES: even some blocks do not need to read the second row, but we still load one to align with other blocks
                        if constexpr (not kMustUseUniformedScaleB)
                            scale_b_1 = ptx::ld_shared(smem_sfb + k_block_idx + shape_k_scales);

                        // Wait for producer WG: B ready (TMA) and A + sfa ready (cp.async)
                        // full_barriers_b[stage_idx]->wait(phase);
                        // DG_DBG_BARRIER("Math:full_b.pass ", stage_idx, full_barriers_b[stage_idx], phase, 0);
                        // full_barriers_a[stage_idx]->wait(phase);
                        // DG_DBG_BARRIER("Math:full_a.pass ", stage_idx, full_barriers_a[stage_idx], phase, 0);
                        full_barriers[stage_idx]->wait(phase);
                        DG_DBG_BARRIER("Math:full.pass ", stage_idx, full_barriers[stage_idx], phase, 0);

                        // TODO: remove some useless computation for unaligned Ms
                        #pragma unroll
                        for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                            auto m_offset = local_idx * WAVE_BLOCK_M;

                            // Read A scales
                            // NOTES: all shared memory read must be prior to `warpgroup_arrive` to avoid next scheduled block polluting the results
                            auto scale_a_0 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_0 + m_offset) : 0;
                            auto scale_a_1 = do_wgmma_store ? ptx::ld_shared(smem_sfa[stage_idx] + r_1 + m_offset) : 0;

                            // Commit WGMMA instructions
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_arrive();
                            #pragma unroll
                            for (uint32_t k = 0; k < BLOCK_K / WGMMA::K; ++ k) {
                                a_desc.reg32_[0] = a_desc_base_lo + (m_offset * BLOCK_K + k * WGMMA::K) / 16;
                                b_desc.reg32_[0] = b_desc_base_lo + k * WGMMA::K / 16;
                                WGMMA::wgmma(a_desc, b_desc, accum, k);
                            }
                            ptx::warpgroup_commit_batch();
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum; ++ i)
                                ptx::warpgroup_fence_operand(accum[i]);
                            ptx::warpgroup_wait<0>();

                            // Notify barrier arrival at the last warpgroup wave
                            if (local_idx == BLOCK_M / WAVE_BLOCK_M - 1)
                                empty_barrier_arrive();

                            // Skip promotion for the unfilled parts
                            if (not do_wgmma_store)
                                continue;

                            // Promote with scales
                            // NOTES: making it as predicates is very important for performance, comparing to two loops
                            float scale_0_0 = scale_a_0 * scale_b_0, scale_1_0 = scale_a_1 * scale_b_0;
                            float scale_0_1, scale_1_1;
                            if constexpr (not kMustUseUniformedScaleB)
                                scale_0_1 = scale_a_0 * scale_b_1, scale_1_1 = scale_a_1 * scale_b_1;

                            auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                            #pragma unroll
                            for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                                // NOTES: for unrolled `num_former_iters` cases, we expect the compiler to automatically make it a constant
                                const bool predicate = kMustUseUniformedScaleB or i < num_former_iters;
                                shifted_accum[i * 4 + 0] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 0];
                                shifted_accum[i * 4 + 1] += (predicate ? scale_0_0 : scale_0_1) * accum[i * 4 + 1];
                                shifted_accum[i * 4 + 2] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 2];
                                shifted_accum[i * 4 + 3] += (predicate ? scale_1_0 : scale_1_1) * accum[i * 4 + 3];
                            }
                        }
                    }
                });
            } else {
                #pragma unroll
                for (uint32_t k_block_idx = 0; k_block_idx < num_total_k_blocks; advance_pipeline(k_block_idx)) {
                    // full_barriers_b[stage_idx]->wait(phase);
                    // DG_DBG_BARRIER("Math(skip):full_b", stage_idx, full_barriers_b[stage_idx], phase, 0);
                    // full_barriers_a[stage_idx]->wait(phase);
                    // DG_DBG_BARRIER("Math(skip):full_a", stage_idx, full_barriers_a[stage_idx], phase, 0);
                    full_barriers[stage_idx]->wait(phase);
                    DG_DBG_BARRIER("Math(skip):full", stage_idx, full_barriers[stage_idx], phase, 0);
                    empty_barrier_arrive();
                }
            }

            // TMA checks
            constexpr uint32_t kNumElemBytes = sizeof(nv_bfloat16);
            constexpr uint32_t TMA_D_BLOCK_N = kSwizzleDMode == 0 ? BLOCK_N : (kSwizzleDMode / kNumElemBytes);
            constexpr uint32_t WGMMA_M_PER_WARP = WGMMA::M / 4;
            DG_STATIC_ASSERT(BLOCK_M % 8 == 0, "Invalid swizzling atom");
            DG_STATIC_ASSERT(BLOCK_N % TMA_D_BLOCK_N == 0 and BLOCK_N / TMA_D_BLOCK_N <= 32,
                            "Unaligned TMA store or too many TMA store instructions");
            DG_STATIC_ASSERT(TMA_D_BLOCK_N % 8 == 0, "Invalid TMA block N");
            constexpr bool kWithGroupOffsetD = kGemmType == GemmType::MGroupedMasked;

            // Skip WGMMA store for the unfilled parts
            if (not do_wgmma_store)
                continue;

            if constexpr (kCombineScatter) {
                const uint32_t base_m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                auto get_scatter_base_and_score = [&](uint32_t logical_m, float& score) -> nv_bfloat16* {
                    score = 0.0f;
                    if (logical_m >= shape_m)
                        return nullptr;
                    const int src_token_i = gather_index == nullptr
                        ? static_cast<int>(logical_m)
                        : __ldg(gather_index + logical_m);
                    const int topk_i = __ldg(combine_row_topk + logical_m);
                    if (src_token_i < 0 or topk_i < 0)
                        return nullptr;
                    if (combine_tokens_per_rank == 0 or
                        static_cast<uint32_t>(topk_i) >= combine_top_k)
                        return nullptr;

                    const uint32_t src_token = static_cast<uint32_t>(src_token_i);
                    const uint32_t src_rank = src_token / combine_tokens_per_rank;
                    if (src_rank >= num_ranks)
                        return nullptr;
                    const uint32_t local_token = src_token - src_rank * combine_tokens_per_rank;
                    const uint64_t peer_base_u64 = __ldg(combine_buffer_ptrs + src_rank);
                    auto* peer_base = reinterpret_cast<nv_bfloat16*>(peer_base_u64);
                    score = __ldg(combine_topk_scores +
                                  static_cast<uint64_t>(src_token) * combine_top_k +
                                  static_cast<uint32_t>(topk_i));
                    const uint64_t dst_row_offset =
                        (static_cast<uint64_t>(local_token) * combine_top_k +
                         static_cast<uint32_t>(topk_i)) * shape_n;
                    return peer_base + dst_row_offset;
                };

                // Stage the WGMMA fragment into a row-major shared-memory tile first.
                // Direct BF16x2 peer stores from accumulator lanes are easy to wire up,
                // but they issue many small global stores. The row-major staging below
                // lets the CTA copy the tile out with 16B vectorized stores instead.
                #pragma unroll
                for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                    const uint32_t m_offset = local_idx * WAVE_BLOCK_M;
                    auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                    #pragma unroll
                    for (uint32_t i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                        auto* smem_ptr = reinterpret_cast<uint8_t*>(
                            smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                        ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                            __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                            __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                            smem_ptr
                        );
                    }
                }
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

                constexpr uint32_t kScatterVecElems = 8;
                DG_STATIC_ASSERT(BLOCK_N % kScatterVecElems == 0, "Invalid vectorized scatter store shape");
                constexpr uint32_t kVecsPerRow = BLOCK_N / kScatterVecElems;

                auto scatter_vec = [&](nv_bfloat16* dst_base, uint32_t row, uint32_t vec, float score) {
                    const uint32_t col = n_block_idx * BLOCK_N + vec * kScatterVecElems;
                    if (dst_base == nullptr or col >= shape_n)
                        return;

                    const auto* src = smem_d + row * BLOCK_N + vec * kScatterVecElems;
                    const uint32_t dst_col = epilogue_type_t::template apply_index_n<kScatterVecElems>(col);
                    if (col + kScatterVecElems <= shape_n) {
                        uint4 packed;
                        auto* packed_bf16 = reinterpret_cast<nv_bfloat16*>(&packed);
                        #pragma unroll
                        for (uint32_t elem = 0; elem < kScatterVecElems; ++elem)
                            packed_bf16[elem] = __float2bfloat16_rn(__bfloat162float(src[elem]) * score);
                        *reinterpret_cast<uint4*>(dst_base + dst_col) = packed;
                    } else {
                        #pragma unroll
                        for (uint32_t elem = 0; elem < kScatterVecElems; ++elem) {
                            if (col + elem < shape_n)
                                dst_base[dst_col + elem] =
                                    __float2bfloat16_rn(__bfloat162float(src[elem]) * score);
                        }
                    }
                };

                for (uint32_t linear = threadIdx.x; linear < BLOCK_M * kVecsPerRow;
                     linear += kNumWGMMAStoreThreads) {
                    const uint32_t row = linear / kVecsPerRow;
                    const uint32_t vec = linear - row * kVecsPerRow;
                    nv_bfloat16* dst_base = nullptr;
                    float score = 0.0f;
                    if constexpr (32 % kVecsPerRow == 0) {
                        const uint32_t leader_lane = lane_idx - (lane_idx % kVecsPerRow);
                        uint32_t dst_base_lo = 0, dst_base_hi = 0;
                        uint32_t score_bits = 0;
                        if (lane_idx == leader_lane) {
                            float leader_score = 0.0f;
                            const uint64_t dst_base_u64 =
                                reinterpret_cast<uint64_t>(
                                    get_scatter_base_and_score(base_m_idx + row, leader_score));
                            dst_base_lo = static_cast<uint32_t>(dst_base_u64);
                            dst_base_hi = static_cast<uint32_t>(dst_base_u64 >> 32);
                            score_bits = __float_as_uint(leader_score);
                        }
                        dst_base_lo = __shfl_sync(0xffffffff, dst_base_lo, leader_lane);
                        dst_base_hi = __shfl_sync(0xffffffff, dst_base_hi, leader_lane);
                        score_bits = __shfl_sync(0xffffffff, score_bits, leader_lane);
                        const uint64_t dst_base_u64 =
                            (static_cast<uint64_t>(dst_base_hi) << 32) | dst_base_lo;
                        dst_base = reinterpret_cast<nv_bfloat16*>(dst_base_u64);
                        score = __uint_as_float(score_bits);
                    } else {
                        dst_base = get_scatter_base_and_score(base_m_idx + row, score);
                    }
                    scatter_vec(dst_base, row, vec, score);
                }
                cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);
                continue;
            }

            // Wait last TMA store to be finished
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N)
                cute::tma_store_wait<0>();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Write back to shared memory using STSM and issue TMA stores
            DG_STATIC_ASSERT(WGMMA::kNumAccum % 4 == 0, "Invalid STSM x2 vectorization");
            #pragma unroll
            for (uint32_t local_idx = 0; local_idx < BLOCK_M / WAVE_BLOCK_M; ++ local_idx) {
                auto m_offset = local_idx * WAVE_BLOCK_M;
                auto shifted_accum = final_accum + WGMMA::kNumAccum * local_idx;
                #pragma unroll
                for (auto i = 0; i < WGMMA::kNumAccum / 4; ++ i) {
                    // Swizzle or padding into the correct address
                    uint8_t* smem_ptr = nullptr;
                    if constexpr (kSwizzleDMode > 0) {
                        // Calculate the swizzling atom offset and in-atom offset
                        constexpr uint32_t kNumBankGroupBytes = 16;
                        auto atom_offset = i / (TMA_D_BLOCK_N / 8), in_atom_offset = i % (TMA_D_BLOCK_N / 8);

                        // Calculate the index of the bank group to be written in the atom
                        auto bank_group_index = in_atom_offset + lane_idx * (kSwizzleDMode / kNumBankGroupBytes);

                        // Reshape the atom in another view and swizzle
                        //  - original: `(BLOCK_M, kSwizzleDMode / kNumBankGroupBytes)`
                        //  - new: `(BLOCK_M * kSwizzleDMode / kNumBankGroupBytes / 8, 8)`
                        constexpr bool kHasShortcut = (kSwizzleDMode / kNumBankGroupBytes) == 8;
                        auto row = kHasShortcut ? (in_atom_offset / 8 + lane_idx) : (bank_group_index / 8);
                        auto col = kHasShortcut ? (in_atom_offset) : (bank_group_index % 8);
                        col ^= row % (kSwizzleDMode / 16);

                        // Add back into the base pointer
                        // NOTES: think twice before modifying this, as changes may affect the number of instructions
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d) +                // Base pointer
                            warp_idx * (WGMMA_M_PER_WARP * kSwizzleDMode) +            // Warp offset
                            m_offset * kSwizzleDMode +                                 // Wave offset
                            atom_offset * BLOCK_M * kSwizzleDMode +                    // Swizzle atom offset (constants)
                            row * (kNumBankGroupBytes * 8) + col * kNumBankGroupBytes; // In-atom offset
                    } else {
                        // No swizzling, just padding
                        smem_ptr = reinterpret_cast<uint8_t*>(smem_d + (m_offset + warp_idx * WGMMA_M_PER_WARP + lane_idx) * BLOCK_N + i * 8);
                    }

                    // NOTES: only 16 lanes' addresses are used
                    ptx::SM90_U32x2_STSM_N<nv_bfloat162>::copy(
                        __float22bfloat162_rn({shifted_accum[i * 4 + 0], shifted_accum[i * 4 + 1]}),
                        __float22bfloat162_rn({shifted_accum[i * 4 + 2], shifted_accum[i * 4 + 3]}),
                        smem_ptr
                    );
                }
            }
            cute::tma_store_fence();
            cutlass::arch::NamedBarrier::sync(kNumWGMMAStoreThreads, 1);

            // Use TMA store to write back to global memory
            // TODO: compatible with FP32 output
            DG_STATIC_ASSERT(kNumWGMMAStoreThreads >= BLOCK_N / TMA_D_BLOCK_N, "Too many TMA blocks");
            if (threadIdx.x < BLOCK_N / TMA_D_BLOCK_N) {
                auto in_block_n_offset = threadIdx.x * TMA_D_BLOCK_N;
                auto smem_ptr = smem_d + in_block_n_offset * BLOCK_M;
                auto n_idx = epilogue_type_t::apply_index_n<TMA_D_BLOCK_N>(n_block_idx * BLOCK_N + in_block_n_offset);
                auto m_idx = scheduler.get_global_idx<kWithGroupOffsetD>(shape_m, BLOCK_M, m_block_idx);
                if constexpr (kGemmType == GemmType::Batched) {
                    cute::SM90_TMA_STORE_3D::copy(&tensor_map_d, smem_ptr,
                                                  n_idx, m_idx, scheduler.current_group_idx);
                } else {
                    cute::SM90_TMA_STORE_2D::copy(&tensor_map_d, smem_ptr, n_idx, m_idx);
                }
                cute::tma_store_arrive();
            }
            __syncwarp();
        }
    }
#else
    if (blockIdx.x == 0 and threadIdx.x == 0)
        DG_DEVICE_ASSERT(false and "This kernel only support sm_90a");
#endif
}

};  // namespace deep_gemm

#pragma clang diagnostic pop
