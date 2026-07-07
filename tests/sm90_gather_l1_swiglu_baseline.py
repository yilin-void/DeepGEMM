"""Standalone SwiGLU quantization used by the SM90 fusion comparison."""

from typing import Tuple

import torch
import triton
import triton.language as tl

try:
    import deep_gemm_moe_L1 as deep_gemm
except ModuleNotFoundError:
    import deep_gemm


@triton.jit
def _swiglu_quant_fp8_kernel(
    x_ptr,
    out_ptr,
    out_sf_ptr,
    shape_m,
    num_tasks,
    stride_x_m: tl.constexpr,
    stride_out_m: tl.constexpr,
    stride_sf_m,
    stride_sf_k,
    INTERMEDIATE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    GRAN_K: tl.constexpr,
):
    task_idx = tl.program_id(0)
    num_n_blocks: tl.constexpr = tl.cdiv(INTERMEDIATE, BLOCK_N)
    groups_per_block: tl.constexpr = BLOCK_N // GRAN_K

    while task_idx < num_tasks:
        m_block = task_idx // num_n_blocks
        n_block = task_idx - m_block * num_n_blocks
        rows = m_block * BLOCK_M + tl.arange(0, BLOCK_M)
        groups = tl.arange(0, groups_per_block)
        elems = tl.arange(0, GRAN_K)
        cols = n_block * BLOCK_N + groups[None, :, None] * GRAN_K + elems[None, None, :]
        mask = (rows[:, None, None] < shape_m) & (cols < INTERMEDIATE)
        offsets = rows[:, None, None] * stride_x_m + cols

        gate = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        up = tl.load(x_ptr + offsets + INTERMEDIATE, mask=mask, other=0.0).to(tl.float32)
        value = gate / (1.0 + tl.exp(-gate)) * up
        amax = tl.max(tl.abs(value), axis=2)
        scale = tl.maximum(amax, 1.0e-4) * (1.0 / 448.0)
        tl.store(out_ptr + rows[:, None, None] * stride_out_m + cols,
                 value / scale[:, :, None], mask=mask)

        scale_cols = n_block * groups_per_block + groups[None, :]
        scale_mask = (rows[:, None] < shape_m) & (scale_cols < INTERMEDIATE // GRAN_K)
        tl.store(out_sf_ptr + rows[:, None] * stride_sf_m + scale_cols * stride_sf_k,
                 scale, mask=scale_mask)
        task_idx += tl.num_programs(0)


def allocate_swiglu_output(shape_m: int, intermediate: int,
                           device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    if shape_m <= 0 or intermediate <= 0 or intermediate % 128 != 0:
        raise ValueError("shape_m must be positive and intermediate must be 128-aligned")
    output = torch.empty((shape_m, intermediate), dtype=torch.float8_e4m3fn, device=device)
    aligned_m = deep_gemm.get_tma_aligned_size(shape_m, 4)
    scales = torch.empty_strided(
        (shape_m, intermediate // 128),
        (1, aligned_m),
        dtype=torch.float32,
        device=device,
    )
    return output, scales


def swiglu_quant_fp8(x: torch.Tensor, output: torch.Tensor,
                     scales: torch.Tensor) -> None:
    if x.dtype != torch.bfloat16 or x.dim() != 2 or not x.is_contiguous():
        raise ValueError("x must be a contiguous BF16 [M, 2I] tensor")
    shape_m, twice_intermediate = x.shape
    if twice_intermediate % 256 != 0:
        raise ValueError("the L1 output dimension must be 256-aligned")
    intermediate = twice_intermediate // 2
    if output.shape != (shape_m, intermediate) or output.dtype != torch.float8_e4m3fn:
        raise ValueError("invalid FP8 output tensor")
    if scales.shape != (shape_m, intermediate // 128) or scales.dtype != torch.float32:
        raise ValueError("invalid FP32 scale tensor")
    if scales.stride(0) != 1:
        raise ValueError("scales must use MN-major layout")

    block_m = 8
    block_n = min(512, triton.next_power_of_2(intermediate))
    num_tasks = triton.cdiv(shape_m, block_m) * triton.cdiv(intermediate, block_n)
    num_sms = torch.cuda.get_device_properties(x.device).multi_processor_count
    _swiglu_quant_fp8_kernel[(min(num_tasks, num_sms * 4),)](
        x,
        output,
        scales,
        shape_m,
        num_tasks,
        x.stride(0),
        output.stride(0),
        scales.stride(0),
        scales.stride(1),
        INTERMEDIATE=intermediate,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        GRAN_K=128,
        num_warps=8,
        num_stages=1,
    )


__all__ = ["allocate_swiglu_output", "swiglu_quant_fp8"]
