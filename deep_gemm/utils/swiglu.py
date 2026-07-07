from typing import Tuple

import torch


QuantizedWeight = Tuple[torch.Tensor, torch.Tensor]

__all__ = ["transform_l1_weights_for_swiglu"]


def transform_l1_weights_for_swiglu(
    weights: QuantizedWeight,
    granularity: int = 8,
) -> QuantizedWeight:
    """Interleave quantized gate/up rows for the SM90 fused L1+SwiGLU path.

    The compact K128 weight-scale tensor stays in its original
    ``[gate blocks, up blocks]`` order. The fused kernel selects the matching
    scale for each interleaved 8-column accumulator atom.
    """
    data, scale = weights
    if data.dtype != torch.float8_e4m3fn or scale.dtype != torch.float32:
        raise TypeError("fused L1+SwiGLU weights require FP8 E4M3 data and FP32 scales")
    if data.dim() != 3 or scale.dim() != 3:
        raise ValueError("weight data and scales must both be rank-3 tensors")
    if not data.is_contiguous() or not scale.is_contiguous():
        raise ValueError("weight data and scales must be contiguous")

    num_experts, twice_intermediate, hidden = data.shape
    if twice_intermediate % 256 != 0 or hidden % 128 != 0:
        raise ValueError("2 * intermediate must be divisible by 256 and hidden by 128")
    if granularity != 8:
        raise ValueError("the SM90 fused epilogue currently requires granularity=8")
    expected_scale_shape = (num_experts, twice_intermediate // 128, hidden // 128)
    if tuple(scale.shape) != expected_scale_shape:
        raise ValueError(
            f"expected weight scale shape {expected_scale_shape}, got {tuple(scale.shape)}"
        )

    intermediate = twice_intermediate // 2
    gate = data[:, :intermediate].reshape(
        num_experts, intermediate // granularity, granularity, hidden
    )
    up = data[:, intermediate:].reshape(
        num_experts, intermediate // granularity, granularity, hidden
    )
    interleaved = torch.stack((gate, up), dim=2).reshape_as(data).contiguous()
    return interleaved, scale
