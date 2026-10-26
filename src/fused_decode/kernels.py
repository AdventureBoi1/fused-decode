"""Triton kernels for the Llama decode regime.

Activations are float16 or bfloat16. Reductions and the GEMV accumulator are
float32. ``ops.py`` decides when these launch: the GEMV is the default only at
M=1, and only through M=16 if the caller forces ``kind``.

``dot`` pads M to 16 and uses tensor cores. ``simt`` keeps the exact M and
issues FMAs. ``dot`` is the default because it ties cuBLAS on the weights that
do not fit in L2. ``simt`` is faster on a single 4096² GEMV that does.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import triton
import triton.language as tl

_SWIGLU_BLOCK = 512
_ROPE_BLOCK = 128
_COPY_BLOCK = 4096


@dataclass(frozen=True)
class RowLayout:
    tensor: torch.Tensor
    a: int
    b: int
    c: int
    sa: int
    sb: int
    sc: int
    col: int
    n: int

    @property
    def rows(self) -> int:
        return self.a * self.b * self.c


def row_layout(t: torch.Tensor) -> RowLayout:
    """Pack up to three leading dimensions. The last dimension is the row."""
    if t.ndim == 0:
        raise ValueError("expected a tensor with a last dimension")
    if any(s < 0 for s in t.stride()):
        t = t.contiguous()
    n = int(t.shape[-1])
    if t.ndim > 4:
        t = t.contiguous().reshape(-1, n)
    if t.ndim == 1:
        return RowLayout(t, 1, 1, 1, 0, 0, 0, int(t.stride(0)), n)
    sizes = [1, 1, 1]
    strides = [0, 0, 0]
    lead = t.ndim - 1
    for i in range(lead):
        sizes[3 - lead + i] = int(t.shape[i])
        strides[3 - lead + i] = int(t.stride(i))
    return RowLayout(
        t,
        sizes[0],
        sizes[1],
        sizes[2],
        strides[0],
        strides[1],
        strides[2],
        int(t.stride(-1)),
        n,
    )


def _same_grid(left: RowLayout, right: RowLayout, what: str) -> None:
    if (left.a, left.b, left.c, left.n) != (right.a, right.b, right.c, right.n):
        raise ValueError(f"{what} cannot be indexed with the same row grid as the input")


def _runnable(layout: RowLayout) -> bool:
    return layout.rows > 0 and layout.n > 0 and layout.b > 0 and layout.c > 0


@triton.jit
def _row_base(row, B, C, sa, sb, sc):
    c = row % C
    t = row // C
    b = t % B
    a = t // B
    return a * sa + b * sb + c * sc


def _rms_warps(block: int) -> int:
    forced = os.environ.get("FUSED_DECODE_RMS_WARPS")
    if forced:
        return int(forced)
    if block <= 1024:
        return 4
    if block <= 4096:
        return 8
    return 16


def _rms_meta(n: int) -> tuple[bool, int, int]:
    forced = os.environ.get("FUSED_DECODE_RMS_BLOCK")
    if forced:
        block = int(forced)
        if block & (block - 1):
            raise ValueError("FUSED_DECODE_RMS_BLOCK must be a power of 2")
        return block >= n, block, _rms_warps(block)
    # Measured on an RTX 4070 Laptop, N=4096, 2048 rows: a 2048-wide
    # two-pass with 8 warps ties the native RMSNorm. A 4096-wide one-pass
    # with 8 warps spills and is ~10x slower.
    if n <= 1024:
        block = max(128, triton.next_power_of_2(n))
        return True, block, 4
    if n <= 2048:
        return True, triton.next_power_of_2(n), 8
    return False, 2048, 8


@triton.jit
def _add_rms_one_pass(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    ro_ptr,
    B,
    C,
    x_sa,
    x_sb,
    x_sc,
    x_col,
    r_sa,
    r_sb,
    r_sc,
    r_col,
    y_sa,
    y_sb,
    y_sc,
    y_col,
    ro_sa,
    ro_sb,
    ro_sc,
    ro_col,
    w_stride,
    N,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x_base = _row_base(row, B, C, x_sa, x_sb, x_sc)
    x = tl.load(x_ptr + x_base + cols * x_col, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        r_base = _row_base(row, B, C, r_sa, r_sb, r_sc)
        x += tl.load(r_ptr + r_base + cols * r_col, mask=mask, other=0.0).to(tl.float32)
    if STORE_RESIDUAL:
        ro_base = _row_base(row, B, C, ro_sa, ro_sb, ro_sc)
        residual = x.to(tl.bfloat16) if OUT_BF16 else x.to(tl.float16)
        tl.store(ro_ptr + ro_base + cols * ro_col, residual, mask=mask)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols * w_stride, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * w
    y_base = _row_base(row, B, C, y_sa, y_sb, y_sc)
    stored = y.to(tl.bfloat16) if OUT_BF16 else y.to(tl.float16)
    tl.store(y_ptr + y_base + cols * y_col, stored, mask=mask)


@triton.jit
def _add_rms_two_pass(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    ro_ptr,
    B,
    C,
    x_sa,
    x_sb,
    x_sc,
    x_col,
    r_sa,
    r_sb,
    r_sc,
    r_col,
    y_sa,
    y_sb,
    y_sc,
    y_col,
    ro_sa,
    ro_sb,
    ro_sc,
    ro_col,
    w_stride,
    N,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    x_base = _row_base(row, B, C, x_sa, x_sb, x_sc)
    r_base = _row_base(row, B, C, r_sa, r_sb, r_sc)
    sumsq = tl.zeros((1,), dtype=tl.float32)
    for start in tl.range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + x_base + cols * x_col, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            x += tl.load(r_ptr + r_base + cols * r_col, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sumsq / N + eps)
    y_base = _row_base(row, B, C, y_sa, y_sb, y_sc)
    ro_base = _row_base(row, B, C, ro_sa, ro_sb, ro_sc)
    for start in tl.range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + x_base + cols * x_col, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            x += tl.load(r_ptr + r_base + cols * r_col, mask=mask, other=0.0).to(tl.float32)
        if STORE_RESIDUAL:
            residual = x.to(tl.bfloat16) if OUT_BF16 else x.to(tl.float16)
            tl.store(ro_ptr + ro_base + cols * ro_col, residual, mask=mask)
        w = tl.load(w_ptr + cols * w_stride, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        stored = y.to(tl.bfloat16) if OUT_BF16 else y.to(tl.float16)
        tl.store(y_ptr + y_base + cols * y_col, stored, mask=mask)


def _rms_launch(layout, strides, kernel, block, warps, n, eps, has_residual, store_residual, out_bf16, pointers):
    x_ptr, r_ptr, w_ptr, y_ptr, ro_ptr, w_stride = pointers
    kernel[(layout.rows,)](
        x_ptr,
        r_ptr,
        w_ptr,
        y_ptr,
        ro_ptr,
        layout.b,
        layout.c,
        *strides["x"],
        *strides["r"],
        *strides["y"],
        *strides["ro"],
        w_stride,
        n,
        eps,
        HAS_RESIDUAL=has_residual,
        STORE_RESIDUAL=store_residual,
        BLOCK=block,
        OUT_BF16=out_bf16,
        num_warps=warps,
    )


def _simple_rows(layout: RowLayout) -> bool:
    return layout.a == 1 and layout.b == 1 and layout.col == 1


@triton.jit
def _rms_simple_one(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    ro_ptr,
    stride_x,
    stride_r,
    stride_y,
    stride_ro,
    w_stride,
    N,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < N
    x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_RESIDUAL:
        x += tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
    if STORE_RESIDUAL:
        residual = x.to(tl.bfloat16) if OUT_BF16 else x.to(tl.float16)
        tl.store(ro_ptr + row * stride_ro + cols, residual, mask=mask)
    inv_rms = tl.rsqrt(tl.sum(x * x, axis=0) / N + eps)
    w = tl.load(w_ptr + cols * w_stride, mask=mask, other=0.0).to(tl.float32)
    y = x * inv_rms * w
    stored = y.to(tl.bfloat16) if OUT_BF16 else y.to(tl.float16)
    tl.store(y_ptr + row * stride_y + cols, stored, mask=mask)


@triton.jit
def _rms_simple_two(
    x_ptr,
    r_ptr,
    w_ptr,
    y_ptr,
    ro_ptr,
    stride_x,
    stride_r,
    stride_y,
    stride_ro,
    w_stride,
    N,
    eps,
    HAS_RESIDUAL: tl.constexpr,
    STORE_RESIDUAL: tl.constexpr,
    BLOCK: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    sumsq = tl.zeros((1,), dtype=tl.float32)
    for start in tl.range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            x += tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        sumsq += tl.sum(x * x, axis=0)
    inv_rms = tl.rsqrt(sumsq / N + eps)
    for start in tl.range(0, N, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < N
        x = tl.load(x_ptr + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
        if HAS_RESIDUAL:
            x += tl.load(r_ptr + row * stride_r + cols, mask=mask, other=0.0).to(tl.float32)
        if STORE_RESIDUAL:
            residual = x.to(tl.bfloat16) if OUT_BF16 else x.to(tl.float16)
            tl.store(ro_ptr + row * stride_ro + cols, residual, mask=mask)
        w = tl.load(w_ptr + cols * w_stride, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        stored = y.to(tl.bfloat16) if OUT_BF16 else y.to(tl.float16)
        tl.store(y_ptr + row * stride_y + cols, stored, mask=mask)


def rms_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    out: torch.Tensor,
    residual: torch.Tensor | None = None,
    residual_out: torch.Tensor | None = None,
) -> None:
    lx = row_layout(x)
    ly = row_layout(out)
    _same_grid(lx, ly, "out")
    has_residual = residual is not None
    store_residual = residual_out is not None
    lr = row_layout(residual) if has_residual else lx
    lro = row_layout(residual_out) if store_residual else ly
    if has_residual:
        _same_grid(lx, lr, "residual")
    if store_residual:
        _same_grid(lx, lro, "residual_out")
    if not _runnable(lx):
        return
    one_pass, block, warps = _rms_meta(lx.n)
    if one_pass and block < lx.n:
        raise RuntimeError(f"one-pass block {block} does not cover N={lx.n}")
    simple = (
        _simple_rows(lx)
        and _simple_rows(ly)
        and (not has_residual or _simple_rows(lr))
        and (not store_residual or _simple_rows(lro))
    )
    if simple:
        kernel = _rms_simple_one if one_pass else _rms_simple_two
        kernel[(lx.rows,)](
            lx.tensor,
            lr.tensor,
            weight,
            ly.tensor,
            lro.tensor,
            lx.sc,
            lr.sc,
            ly.sc,
            lro.sc,
            int(weight.stride(0)),
            lx.n,
            float(eps),
            HAS_RESIDUAL=has_residual,
            STORE_RESIDUAL=store_residual,
            BLOCK=block,
            OUT_BF16=x.dtype == torch.bfloat16,
            num_warps=warps,
        )
        return
    kernel = _add_rms_one_pass if one_pass else _add_rms_two_pass
    strides = {
        "x": (lx.sa, lx.sb, lx.sc, lx.col),
        "r": (lr.sa, lr.sb, lr.sc, lr.col),
        "y": (ly.sa, ly.sb, ly.sc, ly.col),
        "ro": (lro.sa, lro.sb, lro.sc, lro.col),
    }
    _rms_launch(
        lx,
        strides,
        kernel,
        block,
        warps,
        lx.n,
        float(eps),
        has_residual,
        store_residual,
        x.dtype == torch.bfloat16,
        (lx.tensor, lr.tensor, weight, ly.tensor, lro.tensor, int(weight.stride(0))),
    )


@triton.jit
def _swiglu(
    g_ptr,
    u_ptr,
    y_ptr,
    B,
    C,
    g_sa,
    g_sb,
    g_sc,
    g_col,
    u_sa,
    u_sb,
    u_sc,
    u_col,
    y_sa,
    y_sb,
    y_sc,
    y_col,
    N,
    BLOCK: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    g_base = _row_base(row, B, C, g_sa, g_sb, g_sc)
    u_base = _row_base(row, B, C, u_sa, u_sb, u_sc)
    y_base = _row_base(row, B, C, y_sa, y_sb, y_sc)
    g = tl.load(g_ptr + g_base + cols * g_col, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(u_ptr + u_base + cols * u_col, mask=mask, other=0.0).to(tl.float32)
    # Stable sigmoid. tl.sigmoid is 1/(1+exp(-x)) and overflows for large negative x.
    neg = tl.exp(-tl.abs(g))
    sig = tl.where(g >= 0, 1.0 / (1.0 + neg), neg / (1.0 + neg))
    y = g * sig * u
    stored = y.to(tl.bfloat16) if OUT_BF16 else y.to(tl.float16)
    tl.store(y_ptr + y_base + cols * y_col, stored, mask=mask)


def swiglu_out(gate: torch.Tensor, up: torch.Tensor, out: torch.Tensor) -> None:
    lg = row_layout(gate)
    lu = row_layout(up)
    ly = row_layout(out)
    _same_grid(lg, lu, "up")
    _same_grid(lg, ly, "out")
    if not _runnable(lg):
        return
    grid = (lg.rows, triton.cdiv(lg.n, _SWIGLU_BLOCK))
    _swiglu[grid](
        lg.tensor,
        lu.tensor,
        ly.tensor,
        lg.b,
        lg.c,
        lg.sa,
        lg.sb,
        lg.sc,
        lg.col,
        lu.sa,
        lu.sb,
        lu.sc,
        lu.col,
        ly.sa,
        ly.sb,
        ly.sc,
        ly.col,
        lg.n,
        BLOCK=_SWIGLU_BLOCK,
        OUT_BF16=gate.dtype == torch.bfloat16,
        num_warps=4,
    )


@triton.jit
def _rope(
    x_ptr,
    c_ptr,
    s_ptr,
    y_ptr,
    B,
    C,
    x_sa,
    x_sb,
    x_sc,
    x_col,
    c_sa,
    c_sb,
    c_sc,
    c_col,
    s_sa,
    s_sb,
    s_sc,
    s_col,
    y_sa,
    y_sb,
    y_sc,
    y_col,
    N,
    BLOCK: tl.constexpr,
    OUT_BF16: tl.constexpr,
):
    row = tl.program_id(0)
    x_base = _row_base(row, B, C, x_sa, x_sb, x_sc)
    c_base = _row_base(row, B, C, c_sa, c_sb, c_sc)
    s_base = _row_base(row, B, C, s_sa, s_sb, s_sc)
    y_base = _row_base(row, B, C, y_sa, y_sb, y_sc)
    half = N // 2
    for start in tl.range(0, half, BLOCK):
        cols = start + tl.arange(0, BLOCK)
        mask = cols < half
        x1 = tl.load(x_ptr + x_base + cols * x_col, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_ptr + x_base + (cols + half) * x_col, mask=mask, other=0.0).to(tl.float32)
        c1 = tl.load(c_ptr + c_base + cols * c_col, mask=mask, other=0.0).to(tl.float32)
        c2 = tl.load(c_ptr + c_base + (cols + half) * c_col, mask=mask, other=0.0).to(tl.float32)
        s1 = tl.load(s_ptr + s_base + cols * s_col, mask=mask, other=0.0).to(tl.float32)
        s2 = tl.load(s_ptr + s_base + (cols + half) * s_col, mask=mask, other=0.0).to(tl.float32)
        o1 = x1 * c1 - x2 * s1
        o2 = x2 * c2 + x1 * s2
        s1o = o1.to(tl.bfloat16) if OUT_BF16 else o1.to(tl.float16)
        s2o = o2.to(tl.bfloat16) if OUT_BF16 else o2.to(tl.float16)
        tl.store(y_ptr + y_base + cols * y_col, s1o, mask=mask)
        tl.store(y_ptr + y_base + (cols + half) * y_col, s2o, mask=mask)


def rope_out(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, out: torch.Tensor) -> None:
    lx = row_layout(x)
    lc = row_layout(cos)
    ls = row_layout(sin)
    ly = row_layout(out)
    _same_grid(lx, lc, "cos")
    _same_grid(lx, ls, "sin")
    _same_grid(lx, ly, "out")
    if not _runnable(lx):
        return
    _rope[(lx.rows,)](
        lx.tensor,
        lc.tensor,
        ls.tensor,
        ly.tensor,
        lx.b,
        lx.c,
        lx.sa,
        lx.sb,
        lx.sc,
        lx.col,
        lc.sa,
        lc.sb,
        lc.sc,
        lc.col,
        ls.sa,
        ls.sb,
        ls.sc,
        ls.col,
        ly.sa,
        ly.sb,
        ly.sc,
        ly.col,
        lx.n,
        BLOCK=_ROPE_BLOCK,
        OUT_BF16=x.dtype == torch.bfloat16,
        num_warps=4,
    )


def resolve_kind(kind: str | None) -> str:
    if kind is None:
        kind = os.environ.get("FUSED_DECODE_LINEAR_KIND", "dot")
    if kind not in ("dot", "simt"):
        raise ValueError("kind must be 'dot' or 'simt'")
    return kind


def linear_config(kind: str, n: int, k: int) -> tuple[int, int, int, int]:
    """Return BLOCK_N, BLOCK_K, num_warps, num_stages.

    Tiles are multiples of 16 so the dot kernel can hit tensor cores. Override
    with FUSED_DECODE_LINEAR=block_n,block_k,warps,stages while tuning.
    """
    forced = os.environ.get("FUSED_DECODE_LINEAR")
    if forced:
        block_n, block_k, warps, stages = (int(part) for part in forced.split(","))
        return block_n, block_k, warps, stages
    if n >= 2048 and k >= 1024:
        if kind == "dot":
            return 64, 128, 4, 3
        return 64, 64, 4, 3
    return 64, 32, 4, 2


@triton.jit
def _linear_dot(
    x_ptr,
    w_ptr,
    bias_ptr,
    row_ptr,
    y_ptr,
    M,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_bias,
    stride_rm,
    stride_rn,
    stride_ym,
    stride_yn,
    HAS_BIAS: tl.constexpr,
    HAS_ROW: tl.constexpr,
    OUT_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, 16)
    n_mask = offs_n < N
    m_mask = offs_m < M
    acc = tl.zeros((16, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=m_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        # [BLOCK_N, BLOCK_K] keeps K contiguous in a row-major [N, K] weight.
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        )
        if OUT_BF16:
            x = x.to(tl.bfloat16)
            w = w.to(tl.bfloat16)
        else:
            x = x.to(tl.float16)
            w = w.to(tl.float16)
        acc = tl.dot(x, tl.trans(w), acc, out_dtype=tl.float32)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n * stride_bias, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]
    if HAS_ROW:
        extra = tl.load(
            row_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += extra
    stored = acc.to(tl.bfloat16) if OUT_BF16 else acc.to(tl.float16)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        stored,
        mask=m_mask[:, None] & n_mask[None, :],
    )


@triton.jit
def _linear_simt(
    x_ptr,
    w_ptr,
    bias_ptr,
    row_ptr,
    y_ptr,
    N,
    K,
    stride_xm,
    stride_xk,
    stride_wn,
    stride_wk,
    stride_bias,
    stride_rm,
    stride_rn,
    stride_ym,
    stride_yn,
    M: tl.constexpr,
    PAD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    HAS_ROW: tl.constexpr,
    OUT_BF16: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # PAD is the next power of two >= M. tl.arange requires that.
    offs_n = tl.program_id(0) * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, PAD)
    n_mask = offs_n < N
    m_mask = offs_m < M
    acc = tl.zeros((PAD, BLOCK_N), dtype=tl.float32)
    for k0 in tl.range(0, K, BLOCK_K):
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_mask = offs_k < K
        w = tl.load(
            w_ptr + offs_n[:, None] * stride_wn + offs_k[None, :] * stride_wk,
            mask=n_mask[:, None] & k_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        for mi in tl.static_range(M):
            xk = tl.load(
                x_ptr + mi * stride_xm + offs_k * stride_xk,
                mask=k_mask,
                other=0.0,
            ).to(tl.float32)
            partial = tl.sum(w * xk[None, :], axis=1)
            acc += partial[None, :] * (offs_m[:, None] == mi)
    if HAS_BIAS:
        bias = tl.load(bias_ptr + offs_n * stride_bias, mask=n_mask, other=0.0).to(tl.float32)
        acc += bias[None, :]
    if HAS_ROW:
        extra = tl.load(
            row_ptr + offs_m[:, None] * stride_rm + offs_n[None, :] * stride_rn,
            mask=m_mask[:, None] & n_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        acc += extra
    stored = acc.to(tl.bfloat16) if OUT_BF16 else acc.to(tl.float16)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        stored,
        mask=m_mask[:, None] & n_mask[None, :],
    )


def linear_out(
    x: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
    *,
    bias: torch.Tensor | None = None,
    row_add: torch.Tensor | None = None,
    kind: str | None = None,
) -> None:
    """Compute y = x @ weight.T [+ bias] [+ row_add] for x shaped [M, K], M <= 16."""
    kind = resolve_kind(kind)
    m, k = x.shape
    n, k_w = weight.shape
    if k != k_w:
        raise ValueError(f"K mismatch: x has {k}, weight has {k_w}")
    if y.shape != (m, n):
        raise ValueError(f"y shape {tuple(y.shape)} != {(m, n)}")
    if m > 16:
        raise ValueError("linear_out only covers M <= 16")
    if m == 0 or n == 0:
        return
    block_n, block_k, warps, stages = linear_config(kind, n, k)
    if kind == "dot" and (block_n % 16 or block_k % 16 or block_n < 16 or block_k < 16):
        raise ValueError("dot tiles must be multiples of 16 and at least 16")
    bias_ptr = bias if bias is not None else y
    bias_stride = int(bias.stride(0)) if bias is not None else 0
    if row_add is None:
        row_ptr = y
        row_sm, row_sn = 0, 0
    else:
        row_ptr = row_add
        row_sm = 0 if row_add.shape[0] == 1 else int(row_add.stride(0))
        row_sn = 0 if row_add.shape[1] == 1 else int(row_add.stride(1))
    out_bf16 = x.dtype == torch.bfloat16
    grid = (triton.cdiv(n, block_n),)
    if kind == "dot":
        _linear_dot[grid](
            x,
            weight,
            bias_ptr,
            row_ptr,
            y,
            m,
            n,
            k,
            int(x.stride(0)),
            int(x.stride(1)),
            int(weight.stride(0)),
            int(weight.stride(1)),
            bias_stride,
            row_sm,
            row_sn,
            int(y.stride(0)),
            int(y.stride(1)),
            HAS_BIAS=bias is not None,
            HAS_ROW=row_add is not None,
            OUT_BF16=out_bf16,
            BLOCK_N=block_n,
            BLOCK_K=block_k,
            num_warps=warps,
            num_stages=stages,
        )
        return
    _linear_simt[grid](
        x,
        weight,
        bias_ptr,
        row_ptr,
        y,
        n,
        k,
        int(x.stride(0)),
        int(x.stride(1)),
        int(weight.stride(0)),
        int(weight.stride(1)),
        bias_stride,
        row_sm,
        row_sn,
        int(y.stride(0)),
        int(y.stride(1)),
        M=m,
        PAD=triton.next_power_of_2(m),
        HAS_BIAS=bias is not None,
        HAS_ROW=row_add is not None,
        OUT_BF16=out_bf16,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=warps,
        num_stages=stages,
    )


@triton.jit
def _copy_kernel(dst, src, n, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    tl.store(dst + offs, tl.load(src + offs, mask=mask, other=0), mask=mask)


@triton.jit
def _triad_kernel(a, b, c, n, scale, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    av = tl.load(a + offs, mask=mask, other=0)
    bv = tl.load(b + offs, mask=mask, other=0)
    tl.store(c + offs, av + bv * scale, mask=mask)


def launch_copy(dst: torch.Tensor, src: torch.Tensor) -> None:
    n = src.numel()
    grid = (triton.cdiv(n, _COPY_BLOCK),)
    _copy_kernel[grid](dst, src, n, BLOCK=_COPY_BLOCK, num_warps=8)


def launch_triad(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, scale: float = 1.0) -> None:
    n = a.numel()
    grid = (triton.cdiv(n, _COPY_BLOCK),)
    _triad_kernel[grid](a, b, c, n, scale, BLOCK=_COPY_BLOCK, num_warps=8)
