import os
import sys

import pytest
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.insert(0, REPO_ROOT)

try:
    import deep_gemm_moe_L1 as deep_gemm
except ModuleNotFoundError:
    import deep_gemm
    sys.modules["deep_gemm_moe_L1"] = deep_gemm

from generators import (
    MajorTypeAB,
    QuantConfig,
    cast_fp8_fp4_with_major,
    grouped_cast_fp8_fp4_with_major,
)
from sm90_gather_l1_swiglu_baseline import (
    allocate_swiglu_output,
    swiglu_quant_fp8,
)
from deep_gemm_moe_L1.testing import get_arch_major


def _make_problem(hidden: int, intermediate: int, num_experts: int,
                  real_rows_per_expert: int):
    quant = QuantConfig()
    torch.manual_seed(0x51A7 + hidden + intermediate)
    pool_m = num_experts * real_rows_per_expert
    x_bf16 = torch.randn((pool_m, hidden), device="cuda", dtype=torch.bfloat16) * 0.1
    x = cast_fp8_fp4_with_major(
        x_bf16, MajorTypeAB.KMajor, quant.gran_k_a, quant.is_fp4_a, False
    )
    weight_bf16 = torch.randn(
        (num_experts, 2 * intermediate, hidden),
        device="cuda",
        dtype=torch.bfloat16,
    ) * 0.1
    weights = grouped_cast_fp8_fp4_with_major(
        weight_bf16,
        MajorTypeAB.KMajor,
        quant.gran_k_b,
        quant.is_fp4_b,
        False,
        use_block_cast_for_fp8=True,
    )

    shape_m = num_experts * 128
    gather_index = torch.full((shape_m,), -1, device="cuda", dtype=torch.int32)
    permutation = torch.randperm(pool_m, device="cuda", dtype=torch.int64)
    for expert in range(num_experts):
        row_start = expert * 128
        source_start = expert * real_rows_per_expert
        gather_index[row_start:row_start + real_rows_per_expert] = permutation[
            source_start:source_start + real_rows_per_expert
        ].to(torch.int32)
    psum_layout = torch.arange(
        128, shape_m + 1, 128, device="cuda", dtype=torch.int32
    )
    return x, weights, psum_layout, gather_index, shape_m


@pytest.mark.skipif(get_arch_major() != 9, reason="SM90 is required")
@pytest.mark.parametrize(
    "hidden,intermediate,num_experts,real_rows_per_expert",
    [(128, 128, 2, 96), (256, 128, 2, 73), (256, 256, 4, 65)],
)
def test_fused_gather_l1_swiglu_matches_two_kernel_baseline(
    hidden, intermediate, num_experts, real_rows_per_expert
):
    x, weights, psum_layout, gather_index, shape_m = _make_problem(
        hidden, intermediate, num_experts, real_rows_per_expert
    )
    transformed_weights = deep_gemm.transform_l1_weights_for_swiglu(weights)
    kwargs = dict(
        recipe=(1, 128, 128),
        disable_ue8m0_cast=True,
        use_psum_layout=True,
        expected_m_for_psum_layout=128,
        gather_index=gather_index,
    )

    l1_output = torch.empty(
        (shape_m, 2 * intermediate), device="cuda", dtype=torch.bfloat16
    )
    baseline_data, baseline_scale = allocate_swiglu_output(
        shape_m, intermediate, l1_output.device
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        x, weights, l1_output, psum_layout, **kwargs
    )
    swiglu_quant_fp8(l1_output, baseline_data, baseline_scale)

    fused_data, fused_scale = allocate_swiglu_output(
        shape_m, intermediate, l1_output.device
    )
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        x,
        transformed_weights,
        fused_data,
        psum_layout,
        swiglu_output_scale=fused_scale,
        **kwargs,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(fused_scale, baseline_scale, rtol=1e-5, atol=1e-9)
    assert torch.equal(fused_data.float(), baseline_data.float())

    dequant_fused = fused_data.float() * fused_scale.repeat_interleave(128, dim=1)
    dequant_baseline = baseline_data.float() * baseline_scale.repeat_interleave(128, dim=1)
    torch.testing.assert_close(dequant_fused, dequant_baseline, rtol=1e-5, atol=1e-7)

    pad_rows = gather_index < 0
    assert torch.count_nonzero(fused_data[pad_rows].float()).item() == 0


def test_transform_l1_weights_for_swiglu_interleaves_data_only():
    data = torch.arange(2 * 256 * 128, device="cuda", dtype=torch.int32)
    data = (data.remainder(127).float() - 63).to(torch.float8_e4m3fn).reshape(2, 256, 128)
    scale = torch.ones((2, 2, 1), device="cuda", dtype=torch.float32)
    transformed_data, transformed_scale = deep_gemm.transform_l1_weights_for_swiglu(
        (data, scale)
    )
    expected = torch.stack(
        (data[:, :128].reshape(2, 16, 8, 128),
         data[:, 128:].reshape(2, 16, 8, 128)),
        dim=2,
    ).reshape_as(data)
    assert torch.equal(transformed_data.float(), expected.float())
    assert transformed_scale is scale
