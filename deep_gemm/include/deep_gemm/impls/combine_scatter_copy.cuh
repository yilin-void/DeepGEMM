#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads, uint32_t kRowsPerBlock>
__global__ void combine_scatter_copy_rows_kernel(
    const __nv_bfloat16* __restrict__ d_ref,
    const int* __restrict__ gather_index,
    const int* __restrict__ row_to_topk,
    const float* __restrict__ topk_scores,
    const uint64_t* __restrict__ combine_buffer_ptrs,
    uint32_t m,
    uint32_t n,
    uint32_t tokens_per_rank,
    uint32_t top_k,
    uint32_t num_ranks) {
    constexpr uint32_t kVecElems = 8;
    const uint32_t row_base = blockIdx.x * kRowsPerBlock;
    const uint32_t vec_chunks = n / kVecElems;
    const uint32_t tail_start = vec_chunks * kVecElems;

    #pragma unroll
    for (uint32_t row_i = 0; row_i < kRowsPerBlock; ++row_i) {
        const uint32_t row = row_base + row_i;
        if (row >= m)
            return;

        const int src_token_i = gather_index == nullptr
            ? static_cast<int>(row)
            : __ldg(gather_index + row);
        const int topk_i = __ldg(row_to_topk + row);
        if (src_token_i < 0 or topk_i < 0)
            continue;
        if (tokens_per_rank == 0 or static_cast<uint32_t>(topk_i) >= top_k)
            continue;

        const uint32_t src_token = static_cast<uint32_t>(src_token_i);
        const uint32_t src_rank = src_token / tokens_per_rank;
        if (src_rank >= num_ranks)
            continue;

        const uint32_t local_token = src_token - src_rank * tokens_per_rank;
        const uint64_t peer_base_u64 = __ldg(combine_buffer_ptrs + src_rank);
        auto* peer_base = reinterpret_cast<__nv_bfloat16*>(peer_base_u64);
        const float score = __ldg(topk_scores +
                                  static_cast<uint64_t>(src_token) * top_k +
                                  static_cast<uint32_t>(topk_i));
        const uint64_t dst_row_offset =
            (static_cast<uint64_t>(local_token) * top_k + static_cast<uint32_t>(topk_i)) * n;

        const auto* src = d_ref + static_cast<uint64_t>(row) * n;
        auto* dst = peer_base + dst_row_offset;

        for (uint32_t vec = threadIdx.x; vec < vec_chunks; vec += kNumThreads) {
            uint4 packed;
            auto* packed_bf16 = reinterpret_cast<__nv_bfloat16*>(&packed);
            const uint32_t col = vec * kVecElems;
            #pragma unroll
            for (uint32_t elem = 0; elem < kVecElems; ++elem)
                packed_bf16[elem] = __float2bfloat16_rn(__bfloat162float(src[col + elem]) * score);
            reinterpret_cast<uint4*>(dst)[vec] = packed;
        }

        for (uint32_t col = tail_start + threadIdx.x; col < n; col += kNumThreads)
            dst[col] = __float2bfloat16_rn(__bfloat162float(src[col]) * score);
    }
}

}  // namespace deep_gemm
