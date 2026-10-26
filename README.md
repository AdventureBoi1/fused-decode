# fused-decode

Triton kernels for the part of a Llama decode step that is not attention: RMSNorm, RoPE, SwiGLU, and the small-batch GEMV inside the MLP. The benchmark reports latency against stock PyTorch and bandwidth as a fraction of a copy kernel measured on the same GPU.

Attention is not here. FlashAttention already owns the prefill kernel, and decode attention is a different problem (the KV cache). This repo is the other half of the block.

## The regime

A decode step has a tiny batch. For Llama 3 8B that is `M` in 1..8, hidden size 4096, MLP intermediate 14336. The linear layers are matrix-vector products. A 4096×4096 fp16 weight is about 32 MB. The activation next to it is 8 KB. Arithmetic intensity is about 1 FLOP/byte, so the kernel is memory-bound if it is large enough to hide launch overhead, and launch-bound if it is not.

On this machine the GPU is an RTX 4070 Laptop in WDDM. A single RMSNorm of one token moves tens of kilobytes. The launch can cost more than the math. The number that matters for a product is the whole post-attention block, not a one-row microbenchmark. Both are in the results.

## Operators

| Call | What it does |
| --- | --- |
| `rms_norm(x, weight)` | RMSNorm over the last dimension, fp32 variance |
| `add_rms_norm(x, residual, weight)` | `residual + x`, then RMSNorm of the fp32 sum. Pass `residual_out=residual` to update in place |
| `rope(x, cos, sin)` | Llama rotate-half. `cos`/`sin` broadcast like any other PyTorch tensor |
| `swiglu(gate, up)` | `silu(gate) * up` in fp32 |
| `linear_decode(x, weight, bias=None, row_add=None)` | `x @ weight.T`, weight shaped `[N, K]` like `F.linear` |
| `post_attn_mlp(hidden, norm_w, w_gate_up, w_down)` | RMSNorm, stacked gate/up GEMV, SwiGLU, down GEMV with the residual add fused in |

Activations are float16 or bfloat16. There is no backward.

`linear_decode` keeps two GEMV kernels and uses them only where they win.

- `dot` pads `M` to 16 and calls `tl.dot`. Weight traffic does not grow. Issued FLOPs do, by `16/M`.
- `simt` keeps the exact `M` and accumulates fp32 FMAs.

On this GPU the default dispatcher uses `dot` for a flattened batch of 1 and `F.linear` above that. M=2 is already slower than cuBLAS. Pass `kind="dot"` or `kind="simt"` to force a kernel up to 16 rows anyway; that is how the sweep below was measured. `FUSED_DECODE_LINEAR_KIND=simt` changes the default kind. Tile sizes live in `linear_config`. Override them with `FUSED_DECODE_LINEAR=block_n,block_k,warps,stages` and search with `python -m fused_decode.bench --tune`.

## Traffic, so the bandwidth column is auditable

Useful bytes, not bytes after cache reuse:

- RMSNorm: read `x` and `weight`, write `y`. Weight is counted once.
- Fused add + RMSNorm: also read `residual` and write `residual_out`.
- SwiGLU: read `gate`, read `up`, write `out`.
- RoPE: `x` plus `out` plus the stored `cos` and `sin` (not the expanded shape).
- Linear: read `x` and `W`, write `y`.

GB/s uses 1e9. Percent of roof uses the copy kernel in this process, not a datasheet number. A 128-bit-bus estimate from `nvidia-smi`'s max memory clock is recorded next to it for the 4070 Laptop only.

## Reproduce

Windows, from the repo root. PyTorch's CUDA wheel and the matching Triton build are separate installs. Linux can skip `triton-windows`; the PyTorch wheel already depends on Triton.

```
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python -m pytest
.venv\Scripts\python -m fused_decode.bench --compile --out results/rtx4070-laptop.json
```

The bench uses CUDA events. Warmup is not included, so Triton compilation and `torch.compile` are not part of the timed number. `--quick` runs the fp16 headline shapes only.

## Results

Measured on an RTX 4070 Laptop (8 GB, driver 617.14, WDDM) with PyTorch 2.11.0+cu128 and Triton 3.6.0. A streaming copy in this process moves **211 GB/s** of fp16. The memory clock is already at its reported max (8001 MHz); with a 128-bit bus that is **256 GB/s**, so the copy is 82% of that estimate. A 4096³ fp16 GEMM sustains **37 TFLOP/s**. During a loaded matmul the card sat around 75 W and 2550 MHz, under the 140 W cap, so treat absolute times as this laptop's, not a desktop 4070's.

The full sweep is `results/rtx4070-laptop.json`. The rows that decide the design are below. Times are CUDA-event medians. bf16 matches fp16 within a few percent on every row that matters.

| Op | Shape | This repo | PyTorch eager | `torch.compile` |
| --- | --- | --- | --- | --- |
| add + RMSNorm | 8192×4096 fp16 | **1.26 ms** | 1.57 ms | |
| add + RMSNorm | 2048×4096 fp16 | 316 µs | 316 µs | **262 µs** |
| RMSNorm | 8192×4096 fp16 | 634 µs | 627 µs | |
| RMSNorm | 1×4096 fp16 | 96 µs | **46 µs** | |
| SwiGLU | 2048×14336 fp16 | **763 µs** | 1.31 ms | 798 µs |
| SwiGLU | 1×14336 fp16 | 128 µs | **83 µs** | |
| RoPE | 1×2048×32×128 fp16 | 171 µs | 470 µs | **122 µs** |
| RoPE | 1 token, 32 heads | **139 µs** | 279 µs | |
| GEMV, `simt` | M=1, 4096×4096 fp16 | **108 µs** | `mv` 169, `linear` 170 | |
| GEMV, `dot` | M=1, N=4096, K=14336 | 521 µs | `linear` 515, `mv` 514 | |
| GEMV, `dot` | M=1, N=14336, K=4096 | 526 µs | `linear` 512, `mv` 509 | |
| post-attn MLP | M=1, Llama 3 8B widths | 1.57 ms | 1.55 ms | 1.54 ms |
| post-attn MLP | M=8, same widths | 1.64 ms | 1.61 ms | 1.63 ms |

What those rows actually say:

- **Fused add + RMSNorm is the kernel that earns its place.** At 8192 rows the activation does not fit in the 32 MB L2. The fused kernel does the add and the norm at 214 GB/s, on the copy roof, and 1.25× ahead of `x + residual` followed by `F.rms_norm`. Plain RMSNorm at the same size ties the native kernel (634 µs vs 627 µs, also on the roof). One token is the other way around: 96 µs vs 46 µs. That op is a launch, and PyTorch's CUDA kernel is lighter than this Triton launch.
- **SwiGLU is ahead of both eager and Inductor when the tensor is big.** 1.72× against `silu` then a multiply, and still 1.05× against `torch.compile` of that pair. 231 GB/s is 90% of the 256 GB/s clock estimate. One row loses, same reason as RMSNorm: the kernel is smaller than its launch.
- **RoPE beats the eager rotate-half by 2.7× and loses to `torch.compile` (171 µs vs 122 µs).** Inductor fuses the split, the multiply, and the concat into one kernel and does a better job of it than the hand-written one. One token is still 2.0× ahead of eager, because that eager version is many launches.
- **The GEMV matches cuBLAS when the weight does not fit in L2, and beats it when it does.** A 4096×4096 fp16 weight is 32 MB. `simt` finishes it in 108 µs against `torch.mv` at 169 µs. `dot` is 116 µs on the same shape. Its padded MMA issues 4.6 TFLOP/s against a 37 TFLOP/s GEMM roof, so the extra 8 µs is the pad, and the pad is nowhere near compute-bound. The MLP weights are larger: K=14336 or N=14336 is about 112 MB. `dot` lands within 3% of `F.linear` and `torch.mv`, around 225 GB/s. That is the roof. There is no further win hiding in the tile size.
- **M≥2 belongs to cuBLAS.** Forced `dot` at M=4, N=K=4096 takes 146 µs. `F.linear` takes 47 µs. The default dispatcher therefore stops at M=1. Calling `linear_decode` above that is `F.linear` plus the Python checks. Those checks cost tens of microseconds (M=4: 75 µs vs 47 µs) and disappear into the GEMM by M=128 (163 µs vs 169 µs).
- **The MLP block matches PyTorch because the block is the GEMVs.** M=1 is 1.57 ms vs 1.55 ms eager and 1.54 ms compiled. M=8, where the dispatcher has already handed the linears to cuBLAS, is 1.64 ms vs 1.61 ms. Max absolute error against the eager block is 0.002. A CUDA graph of the same block does not move the 1.57 ms. `torch.compile(..., mode="reduce-overhead")` came back at 5.0 ms on this process, so it is not used as a baseline.

`dot` stays the default kind because the shapes that dominate a decode step are the DRAM-bound ones, and there `dot` is the one that ties cuBLAS. `simt` is what you want for the single square GEMV that fits in L2.

## What I would try to upstream

`aten::rms_norm` already exists. The residual add in front of it does not, so a wide activation pays a second kernel and a second trip through memory. `add_rms_norm` is that op. vLLM and Apex already have CUDA versions. The Triton one here is on the copy roof at 8192×4096 and 1.25× ahead of the two-kernel PyTorch version, which is the measurement I would put on the PR.

I would not send the GEMV as a faster cuBLAS. On the weights that do not fit in L2 it ties `torch.mv`. The useful patch is the cutoff: a shape check inside `aten::linear` that calls a GEMV only while it still wins, and the bench that decides the cutoff per architecture. On this laptop that cutoff is M=1. It will not be M=1 on an H100.

I would not upstream `post_attn_mlp`. That is a model fragment, useful here as the integrated measurement.

## Limits

- Inference only. No backward, no custom-op registration, so `torch.compile` of a caller still traces the math unless you wrap these launches yourself.
- RoPE is rotate-half (Llama / GPT-NeoX), not interleaved GPT-J pairs.
- Default `linear_decode` uses the hand-written kernel only at M=1. Passing `kind` forces it up to 16 rows, and above that it is `F.linear`. It does not try to beat cuBLAS on prefill.
- Last-dimension strides are honored for the epilogue kernels. A negative stride is copied first.
- WDDM launch overhead is part of these numbers. A datacenter GPU in TCC will move the crossover.

## Layout

- `src/fused_decode/kernels.py` — Triton kernels and launchers
- `src/fused_decode/ops.py` — checks, the M=16 cutoff, the MLP block
- `src/fused_decode/reference.py` — fp32 reference, linear reference on CPU so TF32 cannot narrow it
- `src/fused_decode/bench.py` — roofline and latency
- `tests/test_ops.py` — correctness, including strided inputs and the cuBLAS fallback
