#pragma once

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <algorithm>
#include <cstring>
#include <nccl.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <sstream>
#include <torch/python.h>

#include "../jit/handle.hpp"
#include "../jit_kernels/impls/combine_reduce.hpp"
#include "../jit_kernels/impls/combine_scatter_check.hpp"
#include "../jit_kernels/impls/combine_scatter_copy.hpp"
#include "../utils/exception.hpp"

namespace deep_gemm::comm {

#ifndef DG_NCCL_CHECK
#define DG_NCCL_CHECK(cmd) \
do { \
    const auto e = (cmd); \
    if (e != ncclSuccess) { \
        std::stringstream ss; \
        ss << static_cast<int>(e) << " (" << ncclGetErrorString(e) << ")"; \
        throw DGException("NCCL", __FILE__, __LINE__, ss.str()); \
    } \
} while (0)
#endif

constexpr const char* kNcclCommCapsuleName = "deep_gemm.nccl_comm";

static ncclComm_t get_nccl_comm_from_capsule(const pybind11::capsule& comm_capsule) {
    auto* ptr = comm_capsule.get_pointer();
    DG_HOST_ASSERT(ptr != nullptr);
    return reinterpret_cast<ncclComm_t>(ptr);
}

static ncclDataType_t get_nccl_dtype(const torch::Tensor& tensor) {
    switch (tensor.scalar_type()) {
        case torch::kFloat:
            return ncclFloat32;
        case torch::kBFloat16:
            return ncclBfloat16;
        case torch::kInt8:
            return ncclInt8;
        default:
            DG_HOST_UNREACHABLE("Unsupported NCCL tensor dtype");
    }
}

static pybind11::bytes nccl_get_unique_id() {
    ncclUniqueId id;
    DG_NCCL_CHECK(ncclGetUniqueId(&id));
    return pybind11::bytes(reinterpret_cast<const char*>(&id), sizeof(id));
}

static pybind11::capsule nccl_comm_init_rank(const pybind11::bytes& unique_id_bytes,
                                             const int& rank,
                                             const int& num_ranks,
                                             const int& device_idx = -1) {
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(rank >= 0 and rank < num_ranks);

    const std::string bytes = unique_id_bytes;
    DG_HOST_ASSERT(bytes.size() == sizeof(ncclUniqueId));
    ncclUniqueId id;
    std::memcpy(&id, bytes.data(), sizeof(id));

    const auto device = device_idx >= 0 ? device_idx : at::cuda::current_device();
    const c10::cuda::CUDAGuard guard(device);
    ncclComm_t comm = nullptr;
    DG_NCCL_CHECK(ncclCommInitRank(&comm, num_ranks, id, rank));
    return pybind11::capsule(reinterpret_cast<void*>(comm), kNcclCommCapsuleName, [](PyObject* capsule) {
        auto* ptr = PyCapsule_GetPointer(capsule, kNcclCommCapsuleName);
        if (ptr != nullptr) {
            (void)ncclCommDestroy(reinterpret_cast<ncclComm_t>(ptr));
        }
    });
}

static void nccl_allgather_bytes(const torch::Tensor& input,
                                 const torch::Tensor& output,
                                 const pybind11::capsule& comm_capsule) {
    DG_HOST_ASSERT(input.is_cuda() and output.is_cuda());
    DG_HOST_ASSERT(input.is_contiguous() and output.is_contiguous());
    DG_HOST_ASSERT(input.device() == output.device());
    DG_HOST_ASSERT(input.nbytes() > 0);

    auto comm = get_nccl_comm_from_capsule(comm_capsule);
    int num_ranks = 0;
    DG_NCCL_CHECK(ncclCommCount(comm, &num_ranks));
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(output.nbytes() == input.nbytes() * static_cast<size_t>(num_ranks));

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream(input.device().index()));
    DG_NCCL_CHECK(ncclAllGather(input.data_ptr(), output.data_ptr(),
                                static_cast<size_t>(input.nbytes()), ncclInt8, comm, stream));
}

static void nccl_reduce_scatter_sum(const torch::Tensor& input,
                                    const torch::Tensor& output,
                                    const pybind11::capsule& comm_capsule) {
    DG_HOST_ASSERT(input.is_cuda() and output.is_cuda());
    DG_HOST_ASSERT(input.is_contiguous() and output.is_contiguous());
    DG_HOST_ASSERT(input.device() == output.device());
    DG_HOST_ASSERT(input.scalar_type() == output.scalar_type());
    DG_HOST_ASSERT(output.numel() > 0);

    auto comm = get_nccl_comm_from_capsule(comm_capsule);
    int num_ranks = 0;
    DG_NCCL_CHECK(ncclCommCount(comm, &num_ranks));
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(input.numel() == output.numel() * static_cast<int64_t>(num_ranks));

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream(input.device().index()));
    DG_NCCL_CHECK(ncclReduceScatter(input.data_ptr(), output.data_ptr(),
                                    static_cast<size_t>(output.numel()), get_nccl_dtype(input),
                                    ncclSum, comm, stream));
}

static double nccl_allgather_bytes_bench(const torch::Tensor& input,
                                         const torch::Tensor& output,
                                         const pybind11::capsule& comm_capsule,
                                         const int& warmups,
                                         const int& iters) {
    DG_HOST_ASSERT(input.is_cuda() and output.is_cuda());
    DG_HOST_ASSERT(input.is_contiguous() and output.is_contiguous());
    DG_HOST_ASSERT(input.device() == output.device());
    DG_HOST_ASSERT(input.nbytes() > 0);
    DG_HOST_ASSERT(warmups >= 0 and iters > 0);

    auto comm = get_nccl_comm_from_capsule(comm_capsule);
    int num_ranks = 0;
    DG_NCCL_CHECK(ncclCommCount(comm, &num_ranks));
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(output.nbytes() == input.nbytes() * static_cast<size_t>(num_ranks));

    const c10::cuda::CUDAGuard guard(input.device());
    const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream(input.device().index()));
    for (int i = 0; i < warmups; ++i) {
        DG_NCCL_CHECK(ncclAllGather(input.data_ptr(), output.data_ptr(),
                                    static_cast<size_t>(input.nbytes()), ncclInt8, comm, stream));
    }
    DG_CUDA_RUNTIME_CHECK(cudaStreamSynchronize(stream));

    cudaEvent_t start = nullptr;
    cudaEvent_t end = nullptr;
    DG_CUDA_RUNTIME_CHECK(cudaEventCreate(&start));
    DG_CUDA_RUNTIME_CHECK(cudaEventCreate(&end));

    std::vector<float> times;
    times.reserve(static_cast<size_t>(iters));
    for (int i = 0; i < iters; ++i) {
        DG_CUDA_RUNTIME_CHECK(cudaEventRecord(start, stream));
        DG_NCCL_CHECK(ncclAllGather(input.data_ptr(), output.data_ptr(),
                                    static_cast<size_t>(input.nbytes()), ncclInt8, comm, stream));
        DG_CUDA_RUNTIME_CHECK(cudaEventRecord(end, stream));
        DG_CUDA_RUNTIME_CHECK(cudaEventSynchronize(end));
        float elapsed_ms = 0.0f;
        DG_CUDA_RUNTIME_CHECK(cudaEventElapsedTime(&elapsed_ms, start, end));
        times.push_back(elapsed_ms);
    }

    DG_CUDA_RUNTIME_CHECK(cudaEventDestroy(start));
    DG_CUDA_RUNTIME_CHECK(cudaEventDestroy(end));

    std::nth_element(times.begin(), times.begin() + times.size() / 2, times.end());
    return static_cast<double>(times[times.size() / 2]);
}

static void check_payloads(const std::vector<torch::Tensor>& inputs,
                           const std::vector<torch::Tensor>& outputs,
                           const int& local_rank,
                           const int& num_ranks) {
    DG_HOST_ASSERT(not inputs.empty());
    DG_HOST_ASSERT(inputs.size() == outputs.size());
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(local_rank >= 0 and local_rank < num_ranks);
    for (size_t i = 0; i < inputs.size(); ++i) {
        DG_HOST_ASSERT(inputs[i].is_cuda() and outputs[i].is_cuda());
        DG_HOST_ASSERT(inputs[i].is_contiguous() and outputs[i].is_contiguous());
        DG_HOST_ASSERT(inputs[i].device() == outputs[i].device());
        DG_HOST_ASSERT(outputs[i].nbytes() % static_cast<size_t>(num_ranks) == 0);
        DG_HOST_ASSERT(outputs[i].nbytes() / static_cast<size_t>(num_ranks) == inputs[i].nbytes());
    }
}

static void check_rank_flags(const torch::Tensor& rank_flags, const int& num_ranks) {
    DG_HOST_ASSERT(rank_flags.is_cuda() and rank_flags.is_contiguous());
    DG_HOST_ASSERT(rank_flags.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(rank_flags.numel() >= num_ranks);
}

static void stream_write_value64(const torch::Tensor& dst, const int64_t& index, const int64_t& value) {
    DG_HOST_ASSERT(dst.is_cuda() and dst.is_contiguous());
    DG_HOST_ASSERT(dst.scalar_type() == torch::kLong);
    DG_HOST_ASSERT(index >= 0 and index < dst.numel());

    const c10::cuda::CUDAGuard guard(dst.device());
    const auto stream = at::cuda::getCurrentCUDAStream(dst.device().index());
    const auto ptr = dst.data_ptr<int64_t>() + index;
    const auto cu_stream = reinterpret_cast<CUstream>(static_cast<cudaStream_t>(stream));
    const auto cu_ptr = static_cast<CUdeviceptr>(reinterpret_cast<uintptr_t>(ptr));
    // Load the v2 symbol explicitly. The unsuffixed legacy export may exist but
    // return CUDA_ERROR_NOT_SUPPORTED on systems where the v2 API works.
    DG_CUDA_DRIVER_CHECK(lazy_cuStreamWriteValue64_v2(cu_stream, cu_ptr, static_cast<cuuint64_t>(value), 0));
}

static void stream_write_value64_ptr(const int64_t& ptr_value, const int64_t& value) {
    DG_HOST_ASSERT(ptr_value != 0);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto cu_stream = reinterpret_cast<CUstream>(static_cast<cudaStream_t>(stream));
    const auto cu_ptr = static_cast<CUdeviceptr>(static_cast<uintptr_t>(ptr_value));
    DG_CUDA_DRIVER_CHECK(lazy_cuStreamWriteValue64_v2(cu_stream, cu_ptr, static_cast<cuuint64_t>(value), 0));
}

static void stream_wait_value64_ptr(const int64_t& ptr_value, const int64_t& value) {
    DG_HOST_ASSERT(ptr_value != 0);

    const auto stream = at::cuda::getCurrentCUDAStream();
    const auto cu_stream = reinterpret_cast<CUstream>(static_cast<cudaStream_t>(stream));
    const auto cu_ptr = static_cast<CUdeviceptr>(static_cast<uintptr_t>(ptr_value));
    DG_CUDA_DRIVER_CHECK(lazy_cuStreamWaitValue64_v2(
        cu_stream, cu_ptr, static_cast<cuuint64_t>(value), CU_STREAM_WAIT_VALUE_GEQ));
}

static pybind11::bytes cuda_ipc_get_mem_handle(const torch::Tensor& tensor) {
    DG_HOST_ASSERT(tensor.is_cuda() and tensor.is_contiguous());
    DG_HOST_ASSERT(tensor.nbytes() > 0);

    const c10::cuda::CUDAGuard guard(tensor.device());
    cudaIpcMemHandle_t handle;
    DG_CUDA_RUNTIME_CHECK(cudaIpcGetMemHandle(&handle, tensor.data_ptr()));
    return pybind11::bytes(reinterpret_cast<const char*>(&handle), sizeof(handle));
}

static torch::Tensor cuda_ipc_alloc_i64(const int64_t& numel) {
    DG_HOST_ASSERT(numel > 0);

    const auto device = at::cuda::current_device();
    const c10::cuda::CUDAGuard guard(device);
    int64_t* ptr = nullptr;
    DG_CUDA_RUNTIME_CHECK(cudaMalloc(&ptr, static_cast<size_t>(numel) * sizeof(int64_t)));
    DG_CUDA_RUNTIME_CHECK(cudaMemset(ptr, 0, static_cast<size_t>(numel) * sizeof(int64_t)));
    auto deleter = [](void* p) {
        if (p != nullptr) {
            const auto error = cudaFree(p);
            DG_HOST_ASSERT(error == cudaSuccess or error == cudaErrorCudartUnloading);
        }
    };
    const auto options = torch::TensorOptions().dtype(torch::kLong).device(torch::kCUDA, device);
    return torch::from_blob(ptr, {numel}, deleter, options);
}

static std::vector<int64_t> cuda_ipc_open_mem_handles(const std::vector<pybind11::bytes>& handles,
                                                       const int& local_rank,
                                                       const torch::Tensor& local_tensor) {
    DG_HOST_ASSERT(local_tensor.is_cuda() and local_tensor.is_contiguous());
    DG_HOST_ASSERT(local_rank >= 0 and local_rank < static_cast<int>(handles.size()));

    const c10::cuda::CUDAGuard guard(local_tensor.device());
    std::vector<int64_t> ptrs(handles.size());
    for (size_t i = 0; i < handles.size(); ++i) {
        if (static_cast<int>(i) == local_rank) {
            ptrs[i] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(local_tensor.data_ptr()));
            continue;
        }

        const std::string bytes = handles[i];
        DG_HOST_ASSERT(bytes.size() == sizeof(cudaIpcMemHandle_t));
        cudaIpcMemHandle_t handle;
        std::memcpy(&handle, bytes.data(), sizeof(handle));
        void* ptr = nullptr;
        DG_CUDA_RUNTIME_CHECK(cudaIpcOpenMemHandle(&ptr, handle, cudaIpcMemLazyEnablePeerAccess));
        ptrs[i] = static_cast<int64_t>(reinterpret_cast<uintptr_t>(ptr));
    }
    return ptrs;
}

static void write_rank_flag_if_needed(const std::optional<torch::Tensor>& rank_flags,
                                      const int& rank,
                                      const int& num_ranks,
                                      const int& value) {
    if (rank_flags.has_value()) {
        check_rank_flags(rank_flags.value(), num_ranks);
        stream_write_value64(rank_flags.value(), rank, value);
    }
}

static void copy_bytes_async(void* dst, const void* src, const size_t& num_bytes, const cudaStream_t& stream) {
    if (num_bytes == 0 or dst == src)
        return;
    DG_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(dst, src, num_bytes, cudaMemcpyDeviceToDevice, stream));
}

static void single_node_allgather_copy_local(const std::vector<torch::Tensor>& inputs,
                                             const std::vector<torch::Tensor>& outputs,
                                             const int& local_rank,
                                             const std::optional<torch::Tensor>& rank_flags = std::nullopt,
                                             const int& flag_value = 1) {
    DG_HOST_ASSERT(not inputs.empty() and inputs.size() == outputs.size());
    DG_HOST_ASSERT(inputs[0].nbytes() > 0);
    DG_HOST_ASSERT(outputs[0].nbytes() % inputs[0].nbytes() == 0);
    const int num_ranks = static_cast<int>(outputs[0].nbytes() / inputs[0].nbytes());
    check_payloads(inputs, outputs, local_rank, num_ranks);
    if (rank_flags.has_value())
        check_rank_flags(rank_flags.value(), num_ranks);

    const c10::cuda::CUDAGuard guard(outputs[0].device());
    const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream(outputs[0].device().index()));
    for (size_t i = 0; i < inputs.size(); ++i) {
        const size_t slot_bytes = inputs[i].nbytes();
        auto* dst = static_cast<uint8_t*>(outputs[i].data_ptr()) + static_cast<size_t>(local_rank) * slot_bytes;
        copy_bytes_async(dst, inputs[i].data_ptr(), slot_bytes, stream);
    }
    write_rank_flag_if_needed(rank_flags, local_rank, num_ranks, flag_value);
}

static void single_node_allgather_pull(const std::vector<torch::Tensor>& outputs,
                                       const std::vector<std::vector<int64_t>>& output_buffer_ptrs,
                                       const int& local_rank,
                                       const int& num_ranks,
                                       const std::optional<torch::Tensor>& rank_flags = std::nullopt,
                                       const int& flag_value = 1) {
    DG_HOST_ASSERT(not outputs.empty());
    DG_HOST_ASSERT(output_buffer_ptrs.size() == outputs.size());
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(local_rank >= 0 and local_rank < num_ranks);
    if (rank_flags.has_value())
        check_rank_flags(rank_flags.value(), num_ranks);

    for (size_t i = 0; i < outputs.size(); ++i) {
        DG_HOST_ASSERT(outputs[i].is_cuda() and outputs[i].is_contiguous());
        DG_HOST_ASSERT(outputs[i].nbytes() % static_cast<size_t>(num_ranks) == 0);
        DG_HOST_ASSERT(static_cast<int>(output_buffer_ptrs[i].size()) >= num_ranks);
    }

    const c10::cuda::CUDAGuard guard(outputs[0].device());
    const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream(outputs[0].device().index()));
    for (int step = 1; step < num_ranks; ++step) {
        const int src_rank = (local_rank + step) % num_ranks;
        for (size_t i = 0; i < outputs.size(); ++i) {
            const size_t slot_bytes = outputs[i].nbytes() / static_cast<size_t>(num_ranks);
            auto* dst = static_cast<uint8_t*>(outputs[i].data_ptr()) + static_cast<size_t>(src_rank) * slot_bytes;
            auto* src_base = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(output_buffer_ptrs[i][src_rank]));
            auto* src = src_base + static_cast<size_t>(src_rank) * slot_bytes;
            copy_bytes_async(dst, src, slot_bytes, stream);
        }
        write_rank_flag_if_needed(rank_flags, src_rank, num_ranks, flag_value);
    }
}

static void single_node_allgather_pull_with_ready_flags(const std::vector<torch::Tensor>& outputs,
                                                        const std::vector<std::vector<int64_t>>& output_buffer_ptrs,
                                                        const std::vector<int64_t>& ready_flag_ptrs,
                                                        const int& local_rank,
                                                        const int& num_ranks,
                                                        const std::optional<torch::Tensor>& rank_flags = std::nullopt,
                                                        const int& flag_value = 1,
                                                        const int64_t& ready_value = 1) {
    DG_HOST_ASSERT(static_cast<int>(ready_flag_ptrs.size()) >= num_ranks);
    DG_HOST_ASSERT(not outputs.empty());
    DG_HOST_ASSERT(output_buffer_ptrs.size() == outputs.size());
    DG_HOST_ASSERT(num_ranks > 0);
    DG_HOST_ASSERT(local_rank >= 0 and local_rank < num_ranks);
    if (rank_flags.has_value())
        check_rank_flags(rank_flags.value(), num_ranks);

    for (size_t i = 0; i < outputs.size(); ++i) {
        DG_HOST_ASSERT(outputs[i].is_cuda() and outputs[i].is_contiguous());
        DG_HOST_ASSERT(outputs[i].nbytes() % static_cast<size_t>(num_ranks) == 0);
        DG_HOST_ASSERT(static_cast<int>(output_buffer_ptrs[i].size()) >= num_ranks);
    }

    const c10::cuda::CUDAGuard guard(outputs[0].device());
    const auto stream = static_cast<cudaStream_t>(at::cuda::getCurrentCUDAStream(outputs[0].device().index()));
    for (int step = 1; step < num_ranks; ++step) {
        const int src_rank = (local_rank + step) % num_ranks;
        stream_wait_value64_ptr(ready_flag_ptrs[src_rank], ready_value);
        for (size_t i = 0; i < outputs.size(); ++i) {
            const size_t slot_bytes = outputs[i].nbytes() / static_cast<size_t>(num_ranks);
            auto* dst = static_cast<uint8_t*>(outputs[i].data_ptr()) + static_cast<size_t>(src_rank) * slot_bytes;
            auto* src_base = reinterpret_cast<uint8_t*>(static_cast<uintptr_t>(output_buffer_ptrs[i][src_rank]));
            auto* src = src_base + static_cast<size_t>(src_rank) * slot_bytes;
            copy_bytes_async(dst, src, slot_bytes, stream);
        }
        write_rank_flag_if_needed(rank_flags, src_rank, num_ranks, flag_value);
    }
}

static void single_node_allgather(const std::vector<torch::Tensor>& inputs,
                                  const std::vector<torch::Tensor>& outputs,
                                  const std::vector<std::vector<int64_t>>& output_buffer_ptrs,
                                  const int& local_rank,
                                  const int& num_ranks,
                                  const pybind11::object& symm_handle,
                                  const std::optional<torch::Tensor>& rank_flags = std::nullopt,
                                  const int& flag_value = 1) {
    single_node_allgather_copy_local(inputs, outputs, local_rank, rank_flags, flag_value);
    DG_HOST_ASSERT(not symm_handle.is_none());
    symm_handle.attr("barrier")();
    single_node_allgather_pull(outputs, output_buffer_ptrs, local_rank, num_ranks, rank_flags, flag_value);
}

static void register_apis(pybind11::module_& m) {
    m.def("nccl_get_unique_id", &nccl_get_unique_id);
    m.def("nccl_comm_init_rank", &nccl_comm_init_rank,
          pybind11::arg("unique_id"), pybind11::arg("rank"),
          pybind11::arg("num_ranks"), pybind11::arg("device_idx") = -1);
    m.def("nccl_allgather_bytes", &nccl_allgather_bytes,
          pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("comm"));
    m.def("nccl_allgather_bytes_bench", &nccl_allgather_bytes_bench,
          pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("comm"),
          pybind11::arg("warmups"), pybind11::arg("iters"));
    m.def("nccl_reduce_scatter_sum", &nccl_reduce_scatter_sum,
          pybind11::arg("input"), pybind11::arg("output"), pybind11::arg("comm"));
    m.def("stream_write_value64", &stream_write_value64,
          pybind11::arg("dst"), pybind11::arg("index"), pybind11::arg("value"));
    m.def("stream_write_value64_ptr", &stream_write_value64_ptr,
          pybind11::arg("ptr"), pybind11::arg("value"));
    m.def("stream_wait_value64_ptr", &stream_wait_value64_ptr,
          pybind11::arg("ptr"), pybind11::arg("value"));
    m.def("cuda_ipc_get_mem_handle", &cuda_ipc_get_mem_handle,
          pybind11::arg("tensor"));
    m.def("cuda_ipc_alloc_i64", &cuda_ipc_alloc_i64,
          pybind11::arg("numel"));
    m.def("cuda_ipc_open_mem_handles", &cuda_ipc_open_mem_handles,
          pybind11::arg("handles"), pybind11::arg("local_rank"), pybind11::arg("local_tensor"));
    m.def("check_combine_scatter_output", &check_combine_scatter_output,
          pybind11::arg("d_ref"), pybind11::arg("gather_index"),
          pybind11::arg("row_to_topk"), pybind11::arg("topk_scores"),
          pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("tokens_per_rank"), pybind11::arg("top_k"),
          pybind11::arg("atol") = 0.0f);
    m.def("combine_scatter_copy_rows", &combine_scatter_copy_rows,
          pybind11::arg("d_ref"), pybind11::arg("gather_index"),
          pybind11::arg("row_to_topk"), pybind11::arg("topk_scores"),
          pybind11::arg("combine_buffer_ptrs"),
          pybind11::arg("tokens_per_rank"), pybind11::arg("top_k"),
          pybind11::arg("rows_per_block") = 4);
    m.def("combine_reduce_slots", &combine_reduce_slots,
          pybind11::arg("combine_buffer"), pybind11::arg("out"));
    m.def("combine_pack_for_reduce_scatter", &combine_pack_for_reduce_scatter,
          pybind11::arg("d_ref"), pybind11::arg("gather_index"),
          pybind11::arg("row_to_topk"), pybind11::arg("topk_scores"),
          pybind11::arg("reduce_scatter_input"),
          pybind11::arg("tokens_per_rank"), pybind11::arg("top_k"));
    m.def("single_node_allgather_copy_local", &single_node_allgather_copy_local,
          pybind11::arg("inputs"), pybind11::arg("outputs"), pybind11::arg("local_rank"),
          pybind11::arg("rank_flags") = std::nullopt, pybind11::arg("flag_value") = 1);
    m.def("single_node_allgather_pull", &single_node_allgather_pull,
          pybind11::arg("outputs"), pybind11::arg("output_buffer_ptrs"),
          pybind11::arg("local_rank"), pybind11::arg("num_ranks"),
          pybind11::arg("rank_flags") = std::nullopt, pybind11::arg("flag_value") = 1);
    m.def("single_node_allgather_pull_with_ready_flags", &single_node_allgather_pull_with_ready_flags,
          pybind11::arg("outputs"), pybind11::arg("output_buffer_ptrs"), pybind11::arg("ready_flag_ptrs"),
          pybind11::arg("local_rank"), pybind11::arg("num_ranks"),
          pybind11::arg("rank_flags") = std::nullopt, pybind11::arg("flag_value") = 1,
          pybind11::arg("ready_value") = 1);
    m.def("single_node_allgather", &single_node_allgather,
          pybind11::arg("inputs"), pybind11::arg("outputs"), pybind11::arg("output_buffer_ptrs"),
          pybind11::arg("local_rank"), pybind11::arg("num_ranks"), pybind11::arg("symm_handle"),
          pybind11::arg("rank_flags") = std::nullopt, pybind11::arg("flag_value") = 1);
}

} // namespace deep_gemm::comm
