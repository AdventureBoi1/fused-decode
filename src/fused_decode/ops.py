"""Public operators.

Inference only: these launches do not record a backward. ``linear_decode``
uses the hand-written kernel at M <= 16 and cuBLAS above that.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from fused_decode.kernels import linear_out, resolve_kind, rms_out, rope_out, swiglu_out

# M=1 beats torch.mv. M=2 is already slower than cuBLAS on an RTX 4070 Laptop,
# so the default dispatcher stops there. Pass kind="dot" or kind="simt" to force
# the hand-written kernel up to 16 rows; linear_out still rejects M > 16.
DECODE_LINEAR_MAX_M = 1
_ACTIVATION = (torch.float16, torch.bfloat16)
_NORM_WEIGHT = (torch.float16, torch.bfloat16, torch.float32)

__all__ = [
    "DECODE_LINEAR_MAX_M",
    "add_rms_norm",
    "linear_decode",
    "post_attn_mlp",
    "rms_norm",
    "rope",
    "swiglu",
]


def _check_activation(t: torch.Tensor, name: str) -> None:
    if not isinstance(t, torch.Tensor):
        raise TypeError(f"{name} must be a tensor")
    if not t.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if t.dtype not in _ACTIVATION:
        raise TypeError(f"{name} must be float16 or bfloat16, got {t.dtype}")


def _check_same(t: torch.Tensor, like: torch.Tensor, name: str) -> None:
    if t.shape != like.shape:
        raise ValueError(f"{name} shape {tuple(t.shape)} != {tuple(like.shape)}")
    if t.dtype != like.dtype:
        raise TypeError(f"{name} dtype {t.dtype} != {like.dtype}")
    if t.device != like.device:
        raise ValueError(f"{name} is on {t.device}, expected {like.device}")


def _check_norm_weight(weight: torch.Tensor, x: torch.Tensor) -> None:
    if weight.device != x.device:
        raise ValueError(f"weight is on {weight.device}, expected {x.device}")
    if weight.dtype not in _NORM_WEIGHT:
        raise TypeError("weight must be float16, bfloat16, or float32")
    if weight.ndim != 1 or weight.shape[0] != x.shape[-1]:
        raise ValueError(f"weight must have shape ({x.shape[-1]},), got {tuple(weight.shape)}")


def _check_eps(eps: float) -> float:
    eps = float(eps)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("eps must be a finite value >= 0")
    return eps


def _check_out(out: torch.Tensor, like: torch.Tensor) -> None:
    _check_same(out, like, "out")


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """RMSNorm over the last dimension. ``weight`` has shape ``(N,)``."""
    _check_activation(x, "x")
    _check_norm_weight(weight, x)
    eps = _check_eps(eps)
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    else:
        _check_out(out, x)
    rms_out(x, weight, eps, out)
    return out


def add_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
    *,
    out: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """``residual_out = x + residual``, then RMSNorm of that sum.

    Pass ``residual_out=residual`` to update the residual stream in place.
    The norm sees the float32 sum, not the rounded stored residual.
    """
    _check_activation(x, "x")
    _check_activation(residual, "residual")
    _check_same(residual, x, "residual")
    _check_norm_weight(weight, x)
    eps = _check_eps(eps)
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    else:
        _check_out(out, x)
    if residual_out is None:
        residual_out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    else:
        _check_same(residual_out, x, "residual_out")
    rms_out(x, weight, eps, out, residual=residual, residual_out=residual_out)
    return out, residual_out


def _broadcast_angles(src: torch.Tensor, like: torch.Tensor, name: str) -> torch.Tensor:
    """Broadcast like PyTorch: shorter shapes align on the right."""
    if src.device != like.device:
        raise ValueError(f"{name} is on {src.device}, expected {like.device}")
    if src.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError(f"{name} must be float16, bfloat16, or float32")
    if src.shape[-1] != like.shape[-1]:
        raise ValueError(f"{name} last dim {src.shape[-1]} != x last dim {like.shape[-1]}")
    try:
        return src.expand(like.shape)
    except RuntimeError as exc:
        raise ValueError(
            f"cannot broadcast {name} shape {tuple(src.shape)} to {tuple(like.shape)}"
        ) from exc


def rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Llama rotate-half RoPE on the last dimension.

    ``cos`` and ``sin`` broadcast over the leading dimensions. This is the
    GPT-NeoX / Llama pairing, not interleaved GPT-J pairs.
    """
    _check_activation(x, "x")
    if x.shape[-1] % 2 != 0:
        raise ValueError("rope last dimension must be even")
    cos_b = _broadcast_angles(cos, x, "cos")
    sin_b = _broadcast_angles(sin, x, "sin")
    if out is None:
        out = torch.empty(x.shape, device=x.device, dtype=x.dtype)
    else:
        _check_out(out, x)
    rope_out(x, cos_b, sin_b, out)
    return out


def swiglu(
    gate: torch.Tensor,
    up: torch.Tensor,
    *,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """``silu(gate) * up``. Same shape, float32 math, stored in ``gate.dtype``."""
    _check_activation(gate, "gate")
    _check_activation(up, "up")
    _check_same(up, gate, "up")
    if out is None:
        out = torch.empty(gate.shape, device=gate.device, dtype=gate.dtype)
    else:
        _check_out(out, gate)
    swiglu_out(gate, up, out)
    return out


def _as_mk(x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...] | None]:
    k = x.shape[-1]
    if x.ndim == 1:
        if not x.is_contiguous():
            x = x.contiguous()
        return x.view(1, k), None
    leading = tuple(x.shape[:-1])
    if x.ndim == 2 and x.stride(0) >= 0 and x.stride(1) >= 0:
        return x, leading
    if not x.is_contiguous():
        x = x.contiguous()
    return x.view(-1, k), leading


def _as_row_add(row_add: torch.Tensor, m: int, n: int, expected: tuple[int, ...]) -> torch.Tensor:
    if row_add.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        raise TypeError("row_add must be float16, bfloat16, or float32")
    if row_add.ndim == 1 and row_add.numel() == n:
        return row_add.view(1, n)
    if row_add.shape == torch.Size(expected) and row_add.is_contiguous():
        return row_add.view(m, n)
    if row_add.ndim == 2 and row_add.shape[0] in (1, m) and row_add.shape[1] in (1, n):
        return row_add
    try:
        return row_add.expand(expected).contiguous().view(m, n)
    except RuntimeError as exc:
        raise ValueError(
            f"row_add shape {tuple(row_add.shape)} is not broadcastable to {expected}"
        ) from exc


def linear_decode(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    *,
    row_add: torch.Tensor | None = None,
    out: torch.Tensor | None = None,
    kind: str | None = None,
) -> torch.Tensor:
    """``y = x @ weight.T + bias``, with an optional epilogue add.

    ``weight`` is ``[N, K]``, matching ``F.linear``. By default the hand-written
    kernel is used only for a flattened batch of 1, which is the shape that
    beats ``torch.mv`` on this GPU. Pass ``kind="dot"`` or ``kind="simt"`` to
    force that kernel up to 16 rows. Wider batches call ``F.linear``.
    """
    _check_activation(x, "x")
    if weight.ndim != 2:
        raise ValueError(f"weight must have shape [N, K], got {tuple(weight.shape)}")
    if weight.device != x.device or weight.dtype != x.dtype:
        raise ValueError("weight device and dtype must match x")
    n, k = weight.shape
    if x.shape[-1] != k:
        raise ValueError(f"x last dim {x.shape[-1]} != weight K {k}")
    # An explicit kind opts into the hand-written kernel through M=16.
    # The default kind stops at DECODE_LINEAR_MAX_M, where it still wins.
    hand_limit = 16 if kind is not None else DECODE_LINEAR_MAX_M
    kind = resolve_kind(kind)
    x2, leading = _as_mk(x)
    m = x2.shape[0]
    expected = (n,) if leading is None else (*leading, n)
    if bias is not None:
        if bias.ndim != 1 or bias.shape[0] != n:
            raise ValueError(f"bias must have shape ({n},), got {tuple(bias.shape)}")
        if bias.device != x.device:
            raise ValueError("bias must be on the same device as x")
        if bias.dtype not in (x.dtype, torch.float32):
            raise TypeError("bias must match x or be float32")
    ra = None
    if row_add is not None:
        if row_add.device != x.device:
            raise ValueError("row_add must be on the same device as x")
        ra = _as_row_add(row_add, m, n, expected)
    if m > hand_limit and out is None:
        produced = F.linear(x2, weight, bias)
        if ra is not None:
            produced = (produced.float() + ra.float()).to(dtype=x.dtype)
        if x.ndim == 1:
            return produced.view(n)
        return produced.view(expected)
    if out is None:
        y_full = torch.empty(expected, device=x.device, dtype=x.dtype)
    else:
        if tuple(out.shape) != expected or out.dtype != x.dtype or out.device != x.device:
            raise ValueError(f"out must be a contiguous {expected} tensor of dtype {x.dtype}")
        if not out.is_contiguous():
            raise ValueError("out must be contiguous")
        y_full = out
    y2 = y_full.view(m, n)
    if m == 0 or n == 0:
        return y_full
    if m > hand_limit:
        produced = F.linear(x2, weight, bias)
        if ra is not None:
            produced = (produced.float() + ra.float()).to(dtype=x.dtype)
        y2.copy_(produced)
        return y_full
    linear_out(x2, weight, y2, bias=bias, row_add=ra, kind=kind)
    return y_full


def post_attn_mlp(
    hidden: torch.Tensor,
    norm_weight: torch.Tensor,
    w_gate_up: torch.Tensor,
    w_down: torch.Tensor,
    eps: float = 1e-6,
    *,
    kind: str | None = None,
) -> torch.Tensor:
    """Llama post-attention block, without the attention itself.

    ``hidden`` is ``[..., H]``. ``w_gate_up`` is ``[2I, H]`` (gate stacked
    above up). ``w_down`` is ``[H, I]``. The residual add is fused into the
    down projection. Returns the block output, which is the next layer's input.
    """
    _check_activation(hidden, "hidden")
    if w_gate_up.ndim != 2 or w_down.ndim != 2:
        raise ValueError("w_gate_up and w_down must be rank-2")
    two_i, h = w_gate_up.shape
    h_down, i = w_down.shape
    if two_i != 2 * i or h != h_down or hidden.shape[-1] != h:
        raise ValueError(
            f"expected w_gate_up [2I, H] and w_down [H, I] with hidden [..., H], "
            f"got {tuple(w_gate_up.shape)}, {tuple(w_down.shape)}, {tuple(hidden.shape)}"
        )
    if w_gate_up.dtype != hidden.dtype or w_down.dtype != hidden.dtype:
        raise TypeError("MLP weights must match hidden dtype")
    if w_gate_up.device != hidden.device or w_down.device != hidden.device:
        raise ValueError("MLP weights must be on the same device as hidden")
    normed = rms_norm(hidden, norm_weight, eps=eps)
    gate_up = linear_decode(normed, w_gate_up, kind=kind)
    gate, up = gate_up.split(i, dim=-1)
    return linear_decode(swiglu(gate, up), w_down, row_add=hidden, kind=kind)
