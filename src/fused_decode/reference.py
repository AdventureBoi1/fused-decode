"""PyTorch references used for correctness.

The linear reference runs in fp32 on the CPU so TF32 cannot narrow it. The
kernels accumulate in fp32 too, then store the activation dtype. Intermediate
results are rounded back to that dtype, matching the kernel pipeline.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    y = xf * torch.rsqrt(var + eps) * weight.float()
    return y.to(dtype=x.dtype)


def add_rms_norm_reference(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    summed = x.float() + residual.float()
    var = summed.pow(2).mean(dim=-1, keepdim=True)
    y = summed * torch.rsqrt(var + eps) * weight.float()
    return y.to(dtype=x.dtype), summed.to(dtype=x.dtype)


def swiglu_reference(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    y = F.silu(gate.float()) * up.float()
    return y.to(dtype=gate.dtype)


def rope_reference(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    cos_b = cos.expand_as(x).float()
    sin_b = sin.expand_as(x).float()
    xf = x.float()
    half = x.shape[-1] // 2
    x1, x2 = xf[..., :half], xf[..., half:]
    c1, c2 = cos_b[..., :half], cos_b[..., half:]
    s1, s2 = sin_b[..., :half], sin_b[..., half:]
    out = torch.cat((x1 * c1 - x2 * s1, x2 * c2 + x1 * s2), dim=-1)
    return out.to(dtype=x.dtype)


def linear_reference(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    row_add: torch.Tensor | None = None,
) -> torch.Tensor:
    y = F.linear(
        x.detach().float().cpu(),
        weight.detach().float().cpu(),
        None if bias is None else bias.detach().float().cpu(),
    )
    if row_add is not None:
        y = y + row_add.detach().float().cpu().reshape_as(y)
    return y.to(device=x.device, dtype=x.dtype)


def post_attn_mlp_reference(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    i = w_down.shape[1]
    h = rms_norm_reference(hidden, norm_weight, eps)
    gu = linear_reference(h, w_gate_up)
    gate, up = gu[..., :i], gu[..., i:]
    mid = swiglu_reference(gate, up)
    y = linear_reference(mid, w_down, row_add=hidden)
    return y
