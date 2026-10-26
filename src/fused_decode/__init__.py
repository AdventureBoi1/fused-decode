"""Decode-regime kernels for a Llama block after attention."""

from fused_decode.ops import (
    DECODE_LINEAR_MAX_M,
    add_rms_norm,
    linear_decode,
    post_attn_mlp,
    rms_norm,
    rope,
    swiglu,
)

__all__ = [
    "DECODE_LINEAR_MAX_M",
    "add_rms_norm",
    "linear_decode",
    "post_attn_mlp",
    "rms_norm",
    "rope",
    "swiglu",
]
__version__ = "0.1.0"
