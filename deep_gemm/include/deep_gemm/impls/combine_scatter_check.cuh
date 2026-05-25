#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <deep_gemm/common/utils.cuh>

namespace deep_gemm {

template <uint32_t kNumThreads>
__global__ void check_combine_scatter_output_kernel(
    const __nv_bfloat16* __restrict__ d_ref,
    const int* __restrict__ gather_index,
    const int* __restrict__ row_to_topk,
    const uint64_t* __restrict__ combine_buffer_ptrs,
    uint32_t m,
    uint32_t n,
    uint32_t tokens_per_rank,
    uint32_t top_k,
    uint32_t num_ranks,
    float atol,
    float* __restrict__ max_abs_diff,
    unsigned long long* __restrict__ mismatch_count) {
    constexpr unsigned int kFloatInfBits = 0x7f800000u;
    const float inf = __uint_as_float(kFloatInfBits);
    const uint64_t total = static_cast<uint64_t>(m) * n;
    for (uint64_t idx = static_cast<uint64_t>(blockIdx.x) * kNumThreads + threadIdx.x;
         idx < total;
         idx += static_cast<uint64_t>(gridDim.x) * kNumThreads) {
        const uint32_t row = static_cast<uint32_t>(idx / n);
        const uint32_t col = static_cast<uint32_t>(idx - static_cast<uint64_t>(row) * n);

        const int src_token_i = __ldg(gather_index + row);
        const int topk_i = __ldg(row_to_topk + row);
        if (src_token_i < 0)
            continue;
        if (topk_i < 0 or tokens_per_rank == 0 or static_cast<uint32_t>(topk_i) >= top_k) {
            atomicAdd(mismatch_count, 1ULL);
            atomicMax(reinterpret_cast<unsigned int*>(max_abs_diff), __float_as_uint(inf));
            continue;
        }

        const uint32_t src_token = static_cast<uint32_t>(src_token_i);
        const uint32_t src_rank = src_token / tokens_per_rank;
        if (src_rank >= num_ranks) {
            atomicAdd(mismatch_count, 1ULL);
            atomicMax(reinterpret_cast<unsigned int*>(max_abs_diff), __float_as_uint(inf));
            continue;
        }

        const uint32_t local_token = src_token - src_rank * tokens_per_rank;
        const uint64_t peer_base_u64 = __ldg(combine_buffer_ptrs + src_rank);
        const auto* peer_base = reinterpret_cast<const __nv_bfloat16*>(peer_base_u64);
        const uint64_t actual_offset =
            (static_cast<uint64_t>(local_token) * top_k + static_cast<uint32_t>(topk_i)) * n + col;

        const float ref = __bfloat162float(d_ref[idx]);
        const float actual = __bfloat162float(peer_base[actual_offset]);
        float diff = fabsf(ref - actual);
        const bool mismatch = not (diff <= atol);
        if (mismatch) {
            if (diff != diff)
                diff = inf;
            atomicAdd(mismatch_count, 1ULL);
            atomicMax(reinterpret_cast<unsigned int*>(max_abs_diff), __float_as_uint(diff));
        }
    }
}

}  // namespace deep_gemm
