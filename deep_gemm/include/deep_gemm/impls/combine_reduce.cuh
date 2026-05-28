#pragma once

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads, uint32_t kTopK, uint32_t kColsPerThread>
__global__ void combine_reduce_slots_kernel(
    const __nv_bfloat16* __restrict__ combine_buffer,
    const float* __restrict__ topk_scores,
    float* __restrict__ out,
    uint32_t tokens_per_rank,
    uint32_t top_k,
    uint32_t n) {
    const uint32_t token = blockIdx.x;
    if (token >= tokens_per_rank)
        return;

    __shared__ float s_scores[kTopK];
    if (threadIdx.x < kTopK)
        s_scores[threadIdx.x] = __ldg(topk_scores + static_cast<uint64_t>(token) * top_k + threadIdx.x);
    __syncthreads();

    constexpr uint32_t kBlockN = kNumThreads * kColsPerThread;
    const uint32_t col_base = blockIdx.y * kBlockN + threadIdx.x;
    float acc[kColsPerThread] = {0.0f};

    #pragma unroll
    for (uint32_t slot = 0; slot < kTopK; ++slot) {
        const float score = s_scores[slot];
        const uint64_t slot_offset =
            (static_cast<uint64_t>(token) * top_k + slot) * n;
        #pragma unroll
        for (uint32_t i = 0; i < kColsPerThread; ++i) {
            const uint32_t col = col_base + i * kNumThreads;
            if (col < n)
                acc[i] += __bfloat162float(combine_buffer[slot_offset + col]) * score;
        }
    }

    const uint64_t out_base = static_cast<uint64_t>(token) * n;
    #pragma unroll
    for (uint32_t i = 0; i < kColsPerThread; ++i) {
        const uint32_t col = col_base + i * kNumThreads;
        if (col < n)
            out[out_base + col] = acc[i];
    }
}

template <uint32_t kNumThreads, uint32_t kTopK, uint32_t kColsPerThread, uint32_t kScaleGroupN>
__global__ void combine_reduce_slots_fp8_kernel(
    const __nv_fp8_e4m3* __restrict__ combine_buffer,
    const float* __restrict__ combine_scales,
    const float* __restrict__ topk_scores,
    float* __restrict__ out,
    uint32_t tokens_per_rank,
    uint32_t top_k,
    uint32_t n) {
    const uint32_t token = blockIdx.x;
    if (token >= tokens_per_rank)
        return;

    __shared__ float s_scores[kTopK];
    if (threadIdx.x < kTopK)
        s_scores[threadIdx.x] = __ldg(topk_scores + static_cast<uint64_t>(token) * top_k + threadIdx.x);
    __syncthreads();

    constexpr uint32_t kBlockN = kNumThreads * kColsPerThread;
    const uint32_t col_base = blockIdx.y * kBlockN + threadIdx.x;
    const uint32_t scale_n = n / kScaleGroupN;
    float acc[kColsPerThread] = {0.0f};

    #pragma unroll
    for (uint32_t slot = 0; slot < kTopK; ++slot) {
        const float score = s_scores[slot];
        const uint64_t slot_offset =
            (static_cast<uint64_t>(token) * top_k + slot) * n;
        const uint64_t scale_slot_offset =
            (static_cast<uint64_t>(token) * top_k + slot) * scale_n;
        #pragma unroll
        for (uint32_t i = 0; i < kColsPerThread; ++i) {
            const uint32_t col = col_base + i * kNumThreads;
            if (col < n) {
                const float scale = __ldg(combine_scales + scale_slot_offset + col / kScaleGroupN);
                const float value = static_cast<float>(combine_buffer[slot_offset + col]) * scale;
                acc[i] += value * score;
            }
        }
    }

    const uint64_t out_base = static_cast<uint64_t>(token) * n;
    #pragma unroll
    for (uint32_t i = 0; i < kColsPerThread; ++i) {
        const uint32_t col = col_base + i * kNumThreads;
        if (col < n)
            out[out_base + col] = acc[i];
    }
}

template <uint32_t kNumThreads>
__global__ void combine_pack_for_reduce_scatter_kernel(
    const __nv_bfloat16* __restrict__ d_ref,
    const int* __restrict__ gather_index,
    const int* __restrict__ row_to_topk,
    const float* __restrict__ topk_scores,
    float* __restrict__ reduce_scatter_input,
    uint32_t m,
    uint32_t n,
    uint32_t tokens_per_rank,
    uint32_t top_k,
    uint32_t num_ranks) {
    const uint64_t total = static_cast<uint64_t>(m) * n;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * kNumThreads + threadIdx.x;
         idx < total;
         idx += static_cast<uint64_t>(gridDim.x) * kNumThreads) {
        const uint32_t row = static_cast<uint32_t>(idx / n);
        const uint32_t col = static_cast<uint32_t>(idx - static_cast<uint64_t>(row) * n);

        const int src_token_i = __ldg(gather_index + row);
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
        const float score = __ldg(topk_scores +
                                  static_cast<uint64_t>(src_token) * top_k +
                                  static_cast<uint32_t>(topk_i));
        const float value = __bfloat162float(d_ref[idx]) * score;
        const uint64_t dst_offset =
            (static_cast<uint64_t>(src_rank) * tokens_per_rank + local_token) * n + col;
        atomicAdd(reduce_scatter_input + dst_offset, value);
    }
}

}  // namespace deep_gemm
