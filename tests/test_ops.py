import pytest
import torch
import torch.nn.functional as F

from fused_decode import (
    add_rms_norm,
    linear_decode,
    post_attn_mlp,
    rms_norm,
    rope,
    swiglu,
)
from fused_decode.reference import (
    add_rms_norm_reference,
    linear_reference,
    post_attn_mlp_reference,
    rms_norm_reference,
    rope_reference,
    swiglu_reference,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DTYPES = [torch.float16, torch.bfloat16]


def _tol(dtype, kind):
    if kind == "linear":
        if dtype == torch.float16:
            return {"rtol": 2e-2, "atol": 2e-2}
        return {"rtol": 4e-2, "atol": 4e-2}
    # fp16 store of a two-pass reduction is about an ulp off the fp32 reference.
    if dtype == torch.float16:
        return {"rtol": 1e-2, "atol": 5e-3}
    return {"rtol": 2e-2, "atol": 2e-2}


def _assert(actual, expected, dtype, kind="pointwise"):
    torch.testing.assert_close(actual, expected, **_tol(dtype, kind))


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(4, 128), (1, 4096), (3, 3072), (5, 100), (2, 8, 64)])
def test_rms_norm(dtype, shape):
    torch.manual_seed(0)
    x = torch.randn(*shape, device="cuda", dtype=dtype)
    w = torch.randn(shape[-1], device="cuda", dtype=dtype)
    _assert(rms_norm(x, w), rms_norm_reference(x, w), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rms_norm_strided_and_fp32_weight(dtype):
    torch.manual_seed(1)
    base = torch.randn(32, 48, device="cuda", dtype=dtype)
    x = base.t()
    assert x.stride(-1) != 1
    w = torch.randn(x.shape[-1], device="cuda", dtype=torch.float32)
    _assert(rms_norm(x, w), rms_norm_reference(x, w), dtype)

    y = torch.randn(7, 9, 32, device="cuda", dtype=dtype).transpose(0, 1)
    assert not y.is_contiguous()
    wy = torch.ones(y.shape[-1], device="cuda", dtype=dtype)
    _assert(rms_norm(y, wy), rms_norm_reference(y, wy), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_add_rms_norm_inplace(dtype):
    torch.manual_seed(2)
    x = torch.randn(6, 256, device="cuda", dtype=dtype)
    residual = torch.randn(6, 256, device="cuda", dtype=dtype)
    w = torch.randn(256, device="cuda", dtype=dtype)
    y, residual_out = add_rms_norm(x, residual, w)
    y_ref, r_ref = add_rms_norm_reference(x, residual, w)
    _assert(y, y_ref, dtype)
    _assert(residual_out, r_ref, dtype)

    residual2 = residual.clone()
    y2, residual_back = add_rms_norm(x, residual2, w, residual_out=residual2)
    assert residual_back.data_ptr() == residual2.data_ptr()
    _assert(y2, y_ref, dtype)
    _assert(residual2, r_ref, dtype)


def test_rms_two_pass():
    torch.manual_seed(3)
    x = torch.randn(2, 9000, device="cuda", dtype=torch.float16)
    w = torch.randn(9000, device="cuda", dtype=torch.float16)
    _assert(rms_norm(x, w), rms_norm_reference(x, w), torch.float16)


@pytest.mark.parametrize("dtype", DTYPES)
def test_rope_broadcast_and_transpose(dtype):
    torch.manual_seed(4)
    x = torch.randn(2, 8, 4, 128, device="cuda", dtype=dtype)
    cos = torch.randn(8, 1, 128, device="cuda", dtype=dtype)
    sin = torch.randn(8, 1, 128, device="cuda", dtype=dtype)
    _assert(rope(x, cos, sin), rope_reference(x, cos, sin), dtype)

    transposed = torch.randn(2, 4, 8, 64, device="cuda", dtype=dtype).transpose(1, 2)
    assert not transposed.is_contiguous()
    cos_t = torch.randn(1, 8, 1, 64, device="cuda", dtype=torch.float32)
    sin_t = torch.randn(1, 8, 1, 64, device="cuda", dtype=torch.float32)
    _assert(rope(transposed, cos_t, sin_t), rope_reference(transposed, cos_t, sin_t), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
def test_swiglu_strided_halves(dtype):
    torch.manual_seed(5)
    rows, inner = 3, 128
    packed = torch.randn(rows, inner * 2, device="cuda", dtype=dtype)
    gate, up = packed.split(inner, dim=-1)
    assert not gate.is_contiguous()
    _assert(swiglu(gate, up), swiglu_reference(gate, up), dtype)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("kind", ["dot", "simt"])
@pytest.mark.parametrize("m,n,k", [(1, 64, 100), (3, 128, 128), (8, 256, 64)])
def test_linear_small(dtype, kind, m, n, k):
    torch.manual_seed(6)
    x = torch.randn(m, k, device="cuda", dtype=dtype) * (k**-0.5)
    w = torch.randn(n, k, device="cuda", dtype=dtype) * (k**-0.5)
    bias = torch.randn(n, device="cuda", dtype=dtype)
    y = linear_decode(x, w, bias, kind=kind)
    _assert(y, linear_reference(x, w, bias), dtype, "linear")


@pytest.mark.parametrize("kind", ["dot", "simt"])
def test_linear_decode_shape_and_strided_weight(kind):
    torch.manual_seed(7)
    dtype = torch.float16
    x = (torch.randn(1, 4096, device="cuda", dtype=dtype) * (4096**-0.5))
    w = torch.randn(1024, 4096, device="cuda", dtype=dtype) * (4096**-0.5)
    _assert(linear_decode(x, w, kind=kind), linear_reference(x, w), dtype, "linear")

    raw = torch.randn(128, 64, device="cuda", dtype=dtype)
    w_t = raw.t()
    assert not w_t.is_contiguous()
    x2 = torch.randn(4, 128, device="cuda", dtype=dtype) * (128**-0.5)
    row = torch.randn(4, 64, device="cuda", dtype=dtype)
    y = linear_decode(x2, w_t, row_add=row, kind=kind)
    _assert(y, linear_reference(x2, w_t, row_add=row), dtype, "linear")


def test_linear_fallback_matches_cublas():
    torch.manual_seed(8)
    x = torch.randn(20, 32, device="cuda", dtype=torch.float16)
    w = torch.randn(48, 32, device="cuda", dtype=torch.float16)
    bias = torch.randn(48, device="cuda", dtype=torch.float16)
    y = linear_decode(x, w, bias)
    torch.testing.assert_close(y, F.linear(x, w, bias), rtol=0, atol=0)

    row = torch.randn(20, 48, device="cuda", dtype=torch.float16)
    fused = linear_decode(x, w, row_add=row)
    manual = F.linear(x, w) + row
    torch.testing.assert_close(fused, manual, rtol=1e-3, atol=1e-3)


def test_post_attn_mlp():
    torch.manual_seed(9)
    dtype = torch.float16
    hidden = 64
    inner = 128
    x = torch.randn(4, hidden, device="cuda", dtype=dtype)
    norm_w = torch.randn(hidden, device="cuda", dtype=dtype)
    w_gu = torch.randn(inner * 2, hidden, device="cuda", dtype=dtype) * (hidden**-0.5)
    w_down = torch.randn(hidden, inner, device="cuda", dtype=dtype) * (inner**-0.5)
    for kind in ("dot", "simt"):
        y = post_attn_mlp(x, norm_w, w_gu, w_down, kind=kind)
        ref = post_attn_mlp_reference(x, norm_w, w_gu, w_down)
        _assert(y, ref, dtype, "linear")


def test_rejected_inputs():
    x = torch.randn(4, 16)
    w = torch.randn(16)
    with pytest.raises(ValueError, match="CUDA"):
        rms_norm(x, w)
    cuda = torch.randn(4, 16, device="cuda", dtype=torch.float16)
    with pytest.raises(ValueError, match="weight"):
        rms_norm(cuda, torch.randn(8, device="cuda", dtype=torch.float16))
    with pytest.raises(ValueError, match="even"):
        rope(
            torch.randn(2, 3, device="cuda", dtype=torch.float16),
            torch.randn(2, 3, device="cuda", dtype=torch.float16),
            torch.randn(2, 3, device="cuda", dtype=torch.float16),
        )
