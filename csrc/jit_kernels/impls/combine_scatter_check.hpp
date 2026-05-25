#pragma once

#include <torch/python.h>

#include "../../jit/compiler.hpp"
#include "../../jit/device_runtime.hpp"
#include "../../jit/kernel_runtime.hpp"
#include "../../utils/exception.hpp"
#include "../../utils/format.hpp"
#include "../../utils/math.hpp"

namespace deep_gemm {

class CombineScatterCheckRuntime final: public LaunchRuntime<CombineScatterCheckRuntime> {
public:
    struct Args {
        void *d_ref, *gather_index, *row_to_topk, *combine_buffer_ptrs;
        uint32_t m, n, tokens_per_rank, top_k, num_ranks;
        float atol;
        void *max_abs_diff, *mismatch_count;
        LaunchArgs launch_args;
    };

    static std::string generate_impl(const Args& args) {
        return fmt::format(R"(
#include <deep_gemm/impls/combine_scatter_check.cuh>

using namespace deep_gemm;

static void __instantiate_kernel() {{
    auto ptr = reinterpret_cast<void*>(&check_combine_scatter_output_kernel<{}>);
}};
)", args.launch_args.num_threads);
    }

    static void launch_impl(const KernelHandle& kernel, const LaunchConfigHandle& config, Args args) {
        DG_CUDA_UNIFIED_CHECK(launch_kernel(kernel, config,
            args.d_ref, args.gather_index, args.row_to_topk, args.combine_buffer_ptrs,
            args.m, args.n, args.tokens_per_rank, args.top_k, args.num_ranks,
            args.atol, args.max_abs_diff, args.mismatch_count));
    }
};

static std::tuple<torch::Tensor, torch::Tensor>
check_combine_scatter_output(const torch::Tensor& d_ref,
                             const torch::Tensor& gather_index,
                             const torch::Tensor& row_to_topk,
                             const torch::Tensor& combine_buffer_ptrs,
                             const int& tokens_per_rank,
                             const int& top_k,
                             const float& atol = 0.0f) {
    DG_HOST_ASSERT(d_ref.is_cuda() and d_ref.is_contiguous());
    DG_HOST_ASSERT(d_ref.scalar_type() == torch::kBFloat16);
    DG_HOST_ASSERT(d_ref.dim() == 2);
    DG_HOST_ASSERT(gather_index.is_cuda() and gather_index.is_contiguous());
    DG_HOST_ASSERT(row_to_topk.is_cuda() and row_to_topk.is_contiguous());
    DG_HOST_ASSERT(combine_buffer_ptrs.is_cuda() and combine_buffer_ptrs.is_contiguous());
    DG_HOST_ASSERT(gather_index.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(row_to_topk.scalar_type() == torch::kInt);
    DG_HOST_ASSERT(combine_buffer_ptrs.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(tokens_per_rank > 0);
    DG_HOST_ASSERT(top_k > 0);

    const auto m64 = d_ref.size(0);
    const auto n64 = d_ref.size(1);
    DG_HOST_ASSERT(m64 >= 0 and m64 <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(n64 > 0 and n64 <= std::numeric_limits<uint32_t>::max());
    DG_HOST_ASSERT(gather_index.numel() >= m64);
    DG_HOST_ASSERT(row_to_topk.numel() >= m64);
    DG_HOST_ASSERT(combine_buffer_ptrs.numel() > 0 and combine_buffer_ptrs.numel() <= 8);

    auto max_abs_diff = torch::zeros({1}, d_ref.options().dtype(torch::kFloat));
    auto mismatch_count = torch::zeros({1}, d_ref.options().dtype(torch::kLong));

    constexpr int num_threads = 256;
    const uint64_t total = static_cast<uint64_t>(m64) * static_cast<uint64_t>(n64);
    const int max_blocks = std::max(1, device_runtime->get_num_sms() * 8);
    const int num_blocks = std::min<int>(max_blocks, static_cast<int>(ceil_div<uint64_t>(total, num_threads)));
    const auto args = CombineScatterCheckRuntime::Args{
        .d_ref = d_ref.data_ptr(),
        .gather_index = gather_index.data_ptr(),
        .row_to_topk = row_to_topk.data_ptr(),
        .combine_buffer_ptrs = combine_buffer_ptrs.data_ptr(),
        .m = static_cast<uint32_t>(m64),
        .n = static_cast<uint32_t>(n64),
        .tokens_per_rank = static_cast<uint32_t>(tokens_per_rank),
        .top_k = static_cast<uint32_t>(top_k),
        .num_ranks = static_cast<uint32_t>(combine_buffer_ptrs.numel()),
        .atol = atol,
        .max_abs_diff = max_abs_diff.data_ptr(),
        .mismatch_count = mismatch_count.data_ptr(),
        .launch_args = LaunchArgs(num_blocks, num_threads),
    };
    const auto code = CombineScatterCheckRuntime::generate(args);
    const auto runtime = compiler->build("combine_scatter_check", code);
    CombineScatterCheckRuntime::launch(runtime, args);
    return std::make_tuple(max_abs_diff, mismatch_count);
}

} // namespace deep_gemm
