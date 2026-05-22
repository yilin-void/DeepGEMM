#pragma once

#include <limits>
#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm {

class CombineReduceSlotsRuntime final: public LaunchRuntime<CombineReduceSlotsRuntime> {
public:
    struct Args {
        void *combine_buffer, *out;
        uint32_t tokens_per_rank, top_k, n;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/combine_reduce.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_reduce_slots_kernel<{}>);
}};
)", args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.combine_buffer, args.out, args.tokens_per_rank, args.top_k, args.n));
    }
};

class CombinePackForReduceScatterRuntime final: public LaunchRuntime<CombinePackForReduceScatterRuntime> {
public:
    struct Args {
        void *d_ref, *gather_index, *row_to_topk, *topk_scores, *reduce_scatter_input;
        uint32_t m, n, tokens_per_rank, top_k, num_ranks;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/combine_reduce.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&combine_pack_for_reduce_scatter_kernel<{}>);
}};
)", args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.d_ref, args.gather_index, args.row_to_topk, args.topk_scores, args.reduce_scatter_input,
            args.m, args.n, args.tokens_per_rank, args.top_k, args.num_ranks));
    }
};

static void combine_reduce_slots(const torch::Tensor& combine_buffer,
                                 const torch::Tensor& out) {
    DG_HOST_ASSERT(combine_buffer.is_cuda() and combine_buffer.is_contiguous());
    DG_HOST_ASSERT(out.is_cuda() and out.is_contiguous());
    DG_HOST_ASSERT(combine_buffer.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(out.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(combine_buffer.dim() == 3);
    DG_HOST_ASSERT(out.dim() == 2);

    const auto tokens64 = combine_buffer.size(0);
    const auto top_k64 = combine_buffer.size(1);
    const auto n64 = combine_buffer.size(2);
    DG_HOST_ASSERT(tokens64 >= 0 and tokens64 <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(top_k64 > 0 and top_k64 <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(n64 > 0 and n64 <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(out.size(0) == tokens64 and out.size(1) == n64);

    constexpr int num_threads = 256;
    const uint64_t total = static_cast<uint64_t>(tokens64) * static_cast<uint64_t>(n64);
    const int max_blocks = std::max(1, device_runtime->get_num_sms() * 8);
    const int num_blocks = std::min<int>(max_blocks, static_cast<int>(ceil_div<uint64_t>(total, num_threads)));
    const auto args = CombineReduceSlotsRuntime::Args{
        .combine_buffer = combine_buffer.data_ptr(),
        .out = out.data_ptr(),
        .tokens_per_rank = static_cast<uint32_t>(tokens64),
        .top_k = static_cast<uint32_t>(top_k64),
        .n = static_cast<uint32_t>(n64),
        .launch_args = LaunchArgs(num_blocks, num_threads),
    };
    const auto code = CombineReduceSlotsRuntime::generate(args);
    const auto runtime = compiler->build("combine_reduce_slots", code);
    CombineReduceSlotsRuntime::launch(runtime, args);
}

static void combine_pack_for_reduce_scatter(const torch::Tensor& d_ref,
                                            const torch::Tensor& gather_index,
                                            const torch::Tensor& row_to_topk,
                                            const torch::Tensor& topk_scores,
                                            const torch::Tensor& reduce_scatter_input,
                                            const int& tokens_per_rank,
                                            const int& top_k) {
    DG_HOST_ASSERT(d_ref.is_cuda() and d_ref.is_contiguous());
    DG_HOST_ASSERT(gather_index.is_cuda() and gather_index.is_contiguous());
    DG_HOST_ASSERT(row_to_topk.is_cuda() and row_to_topk.is_contiguous());
    DG_HOST_ASSERT(topk_scores.is_cuda() and topk_scores.is_contiguous());
    DG_HOST_ASSERT(reduce_scatter_input.is_cuda() and reduce_scatter_input.is_contiguous());
    DG_HOST_ASSERT(d_ref.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(gather_index.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(row_to_topk.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(topk_scores.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(reduce_scatter_input.scalar_type() == torch::kFloat);
    DG_HOST_ASSERT(d_ref.dim() == 2);
    DG_HOST_ASSERT(topk_scores.dim() == 2);
    DG_HOST_ASSERT(reduce_scatter_input.dim() == 3);
    DG_HOST_ASSERT(tokens_per_rank > 0);
    DG_HOST_ASSERT(top_k > 0);

    const auto m64 = d_ref.size(0);
    const auto n64 = d_ref.size(1);
    const auto num_ranks64 = reduce_scatter_input.size(0);
    DG_HOST_ASSERT(m64 >= 0 and m64 <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(n64 > 0 and n64 <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(num_ranks64 > 0 and num_ranks64 <= 8);
    DG_HOST_ASSERT(reduce_scatter_input.size(1) == tokens_per_rank);
    DG_HOST_ASSERT(reduce_scatter_input.size(2) == n64);
    DG_HOST_ASSERT(gather_index.numel() >= m64);
    DG_HOST_ASSERT(row_to_topk.numel() >= m64);
    DG_HOST_ASSERT(topk_scores.size(0) >= static_cast<int64_t>(tokens_per_rank) * num_ranks64);
    DG_HOST_ASSERT(topk_scores.size(1) >= top_k);

    constexpr int num_threads = 256;
    const uint64_t total = static_cast<uint64_t>(m64) * static_cast<uint64_t>(n64);
    const int max_blocks = std::max(1, device_runtime->get_num_sms() * 8);
    const int num_blocks = std::min<int>(max_blocks, static_cast<int>(ceil_div<uint64_t>(total, num_threads)));
    const auto args = CombinePackForReduceScatterRuntime::Args{
        .d_ref = d_ref.data_ptr(),
        .gather_index = gather_index.data_ptr(),
        .row_to_topk = row_to_topk.data_ptr(),
        .topk_scores = topk_scores.data_ptr(),
        .reduce_scatter_input = reduce_scatter_input.data_ptr(),
        .m = static_cast<uint32_t>(m64),
        .n = static_cast<uint32_t>(n64),
        .tokens_per_rank = static_cast<uint32_t>(tokens_per_rank),
        .top_k = static_cast<uint32_t>(top_k),
        .num_ranks = static_cast<uint32_t>(num_ranks64),
        .launch_args = LaunchArgs(num_blocks, num_threads),
    };
    const auto code = CombinePackForReduceScatterRuntime::generate(args);
    const auto runtime = compiler->build("combine_pack_for_reduce_scatter", code);
    CombinePackForReduceScatterRuntime::launch(runtime, args);
}

} // namespace deep_gemm
