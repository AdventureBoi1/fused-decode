"""Roofline and latency measurements.

Run from the repo root:

    python -m fused_decode.bench --out results/rtx4070-laptop.json
    python -m fused_decode.bench --compile --out results/rtx4070-laptop.json
    python -m fused_decode.bench --tune

Timings are CUDA-event medians. Warmup is excluded, so Triton and
torch.compile compilation are not part of the number. Bandwidth is useful
bytes divided by that time, using decimal GB (1e9). The percent-of-roof
column uses the measured copy kernel on this GPU, not a datasheet figure.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import threading
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from fused_decode.kernels import launch_copy, launch_triad, linear_out
from fused_decode.ops import add_rms_norm, linear_decode, post_attn_mlp, rms_norm, rope, swiglu
from fused_decode.reference import linear_reference, rope_reference

LLAMA_H = 4096
LLAMA_I = 14336
_ROOF_ELEMS = 64 * 1024 * 1024


def _sync() -> None:
    torch.cuda.synchronize()


def time_us(fn, iters: int, warmup: int = 10, rounds: int = 5) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    samples = []
    for _ in range(rounds):
        start.record()
        for _ in range(iters):
            fn()
        end.record()
        _sync()
        samples.append(start.elapsed_time(end) / iters * 1000.0)
    return statistics.median(samples)


def _gpu_info() -> dict:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total,clocks.max.sm,clocks.max.mem,power.max_limit",
        "--format=csv,noheader,nounits",
    ]
    text = subprocess.check_output(cmd, text=True).strip()
    parts = [part.strip() for part in text.split(",")]
    keys = ["name", "driver", "memory_mib", "max_sm_mhz", "max_mem_mhz", "max_power_w"]
    info = dict(zip(keys, parts))
    if "4070 Laptop" in info.get("name", ""):
        mem_mhz = float(info["max_mem_mhz"])
        # nvidia-smi reports the DDR command clock. Data rate is 2x, bus is 128-bit.
        info["theoretical_gbs"] = mem_mhz * 1e6 * 2 * 128 / 8 / 1e9
        info["theoretical_gbs_note"] = "mem_clock_mhz * 2 * 128-bit / 8, from nvidia-smi max memory clock"
    return info


def _sample_clocks() -> dict:
    a = torch.empty(4096, 4096, device="cuda", dtype=torch.float16).normal_()
    b = torch.empty_like(a)
    samples = []
    stop = threading.Event()

    def poll() -> None:
        while not stop.is_set():
            try:
                out = subprocess.check_output(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw,clocks.sm,clocks.mem,utilization.gpu",
                        "--format=csv,noheader,nounits",
                    ],
                    text=True,
                    timeout=2,
                ).strip()
                samples.append(out)
            except (subprocess.SubprocessError, OSError):
                pass
            stop.wait(0.2)

    thread = threading.Thread(target=poll, daemon=True)
    thread.start()
    deadline = time.perf_counter() + 1.5
    while time.perf_counter() < deadline:
        torch.mm(a, b)
    _sync()
    stop.set()
    thread.join(timeout=2)
    parsed = []
    for line in samples:
        nums = []
        ok = True
        for piece in line.split(","):
            try:
                nums.append(float(piece.strip()))
            except ValueError:
                ok = False
        if ok and len(nums) >= 4:
            parsed.append(nums)
    if not parsed:
        return {}
    cols = list(zip(*parsed))
    return {
        "power_w": statistics.median(cols[0]),
        "sm_mhz": statistics.median(cols[1]),
        "mem_mhz": statistics.median(cols[2]),
        "util_pct": statistics.median(cols[3]),
        "samples": len(parsed),
    }


def _speedup(eager_us, ours_us):
    if not eager_us or not ours_us:
        return None
    return eager_us / ours_us


def bench_roofline(dtype: torch.dtype) -> dict:
    n = _ROOF_ELEMS
    a = torch.empty(n, device="cuda", dtype=dtype)
    b = torch.empty(n, device="cuda", dtype=dtype)
    c = torch.empty(n, device="cuda", dtype=dtype)
    a.normal_()
    b.normal_()
    item = a.element_size()
    copy_us = time_us(lambda: launch_copy(c, a), iters=20, warmup=5)
    triad_us = time_us(lambda: launch_triad(a, b, c, 1.25), iters=20, warmup=5)
    side = 4096
    left = torch.empty(side, side, device="cuda", dtype=dtype).normal_()
    right = torch.empty(side, side, device="cuda", dtype=dtype).normal_()
    mm_us = time_us(lambda: torch.mm(left, right), iters=10, warmup=5, rounds=3)
    flops = 2.0 * side**3
    return {
        "dtype": str(dtype).replace("torch.", ""),
        "copy_gbs": (2 * n * item) / (copy_us * 1e-6) / 1e9,
        "triad_gbs": (3 * n * item) / (triad_us * 1e-6) / 1e9,
        "copy_us": copy_us,
        "triad_us": triad_us,
        "cublas_tflops": flops / (mm_us * 1e-6) / 1e12,
        "cublas_mm_us": mm_us,
        "cublas_mm_n": side,
    }


def _pointwise_row(name, fn, eager, nbytes, roof_gbs, check=None, compile_eager=False) -> dict:
    iters = 100 if nbytes < 1_000_000 else 30
    ours_us = time_us(fn, iters=iters)
    eager_us = time_us(eager, iters=iters)
    seconds = ours_us * 1e-6
    gbs = nbytes / seconds / 1e9
    row = {
        "name": name,
        "us": ours_us,
        "eager_us": eager_us,
        "speedup_vs_eager": _speedup(eager_us, ours_us),
        "gbs": gbs,
        "pct_copy_roof": (gbs / roof_gbs * 100.0) if roof_gbs else None,
        "bytes": nbytes,
    }
    if check is not None:
        row["max_abs"] = check()
    if compile_eager:
        try:
            compiled = torch.compile(eager, mode="default", dynamic=False)
            compiled_us = time_us(compiled, iters=max(10, iters // 2), warmup=3, rounds=3)
            row["compile_default_us"] = compiled_us
            row["speedup_vs_compile_default"] = _speedup(compiled_us, ours_us)
        except Exception as exc:
            row["compile_default_error"] = f"{type(exc).__name__}: {exc}"
            print("compile failed", name, exc)
    return row


def bench_epilogue(dtype: torch.dtype, roof_gbs: float, quick: bool, compile_eager: bool) -> list[dict]:
    rows = []
    item = torch.empty((), dtype=dtype).element_size()
    # 8192 rows is 64MB and does not fit in the 32MB L2. 2048 rows does.
    shapes = [(2048, LLAMA_H), (1, LLAMA_H)] if quick else [(8192, LLAMA_H), (2048, LLAMA_H), (1, LLAMA_H)]
    for rows_n, hidden in shapes:
        x = torch.randn(rows_n, hidden, device="cuda", dtype=dtype)
        residual = torch.randn_like(x)
        weight = torch.randn(hidden, device="cuda", dtype=dtype)

        def run_rms(x=x, weight=weight):
            return rms_norm(x, weight)

        def eager_rms(x=x, weight=weight):
            return F.rms_norm(x, (hidden,), weight, 1e-6)

        def run_add(x=x, residual=residual, weight=weight):
            return add_rms_norm(x, residual, weight)

        def eager_add(x=x, residual=residual, weight=weight):
            return F.rms_norm(x + residual, (hidden,), weight, 1e-6)

        rms_bytes = (rows_n * hidden + hidden + rows_n * hidden) * item
        add_bytes = (3 * rows_n * hidden + hidden + rows_n * hidden) * item
        rows.append(
            _pointwise_row(
                f"rms_norm {rows_n}x{hidden} {dtype}",
                run_rms,
                eager_rms,
                rms_bytes,
                roof_gbs,
                check=lambda x=x, weight=weight: (rms_norm(x, weight) - F.rms_norm(x, (hidden,), weight, 1e-6))
                .abs()
                .max()
                .item(),
            )
        )
        rows.append(
            _pointwise_row(
                f"add_rms_norm {rows_n}x{hidden} {dtype}",
                run_add,
                eager_add,
                add_bytes,
                roof_gbs,
                compile_eager=compile_eager and dtype == torch.float16 and rows_n == 2048,
            )
        )
        print(rows[-2]["name"], f"{rows[-2]['us']:.2f} us", f"eager {rows[-2]['eager_us']:.2f}")
        print(rows[-1]["name"], f"{rows[-1]['us']:.2f} us", f"eager {rows[-1]['eager_us']:.2f}")

    swiglu_shapes = [(1, LLAMA_I), (2048, LLAMA_I)] if not quick else [(2048, LLAMA_I)]
    for rows_n, inner in swiglu_shapes:
        gate = torch.randn(rows_n, inner, device="cuda", dtype=dtype)
        up = torch.randn_like(gate)
        nbytes = 3 * gate.numel() * item

        def run(gate=gate, up=up):
            return swiglu(gate, up)

        def eager(gate=gate, up=up):
            return F.silu(gate) * up

        rows.append(
            _pointwise_row(
                f"swiglu {rows_n}x{inner} {dtype}",
                run,
                eager,
                nbytes,
                roof_gbs,
                compile_eager=compile_eager and dtype == torch.float16 and rows_n == 2048,
            )
        )
        print(rows[-1]["name"], f"{rows[-1]['us']:.2f} us", f"eager {rows[-1]['eager_us']:.2f}")

    rope_shapes = [((1, 2048, 32, 128), (2048, 1, 128))]
    if not quick:
        rope_shapes.append(((1, 1, 32, 128), (1, 1, 128)))
    for x_shape, cos_shape in rope_shapes:
        x = torch.randn(*x_shape, device="cuda", dtype=dtype)
        cos = torch.randn(*cos_shape, device="cuda", dtype=dtype)
        sin = torch.randn(*cos_shape, device="cuda", dtype=dtype)
        nbytes = x.nbytes * 2 + cos.nbytes + sin.nbytes

        def run(x=x, cos=cos, sin=sin):
            return rope(x, cos, sin)

        def eager(x=x, cos=cos, sin=sin):
            cos_b = cos.expand_as(x)
            sin_b = sin.expand_as(x)
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            c1, c2 = cos_b[..., :half], cos_b[..., half:]
            s1, s2 = sin_b[..., :half], sin_b[..., half:]
            return torch.cat((x1 * c1 - x2 * s1, x2 * c2 + x1 * s2), dim=-1)

        rows.append(
            _pointwise_row(
                f"rope {tuple(x_shape)} {dtype}",
                run,
                eager,
                nbytes,
                roof_gbs,
                check=lambda x=x, cos=cos, sin=sin: (rope(x, cos, sin) - rope_reference(x, cos, sin)).abs().max().item(),
                compile_eager=compile_eager and dtype == torch.float16 and x_shape[1] == 2048,
            )
        )
        print(rows[-1]["name"], f"{rows[-1]['us']:.2f} us", f"eager {rows[-1]['eager_us']:.2f}")
    return rows


def _linear_nbytes(m, n, k, dtype, bias=False):
    item = torch.empty((), dtype=dtype).element_size()
    return (m * k + n * k + m * n + (n if bias else 0)) * item


def bench_linear(dtype: torch.dtype, roof_gbs: float, cublas_tflops: float, quick: bool) -> list[dict]:
    rows = []
    shapes = [(1, LLAMA_H, LLAMA_H), (8, LLAMA_H, LLAMA_H), (32, LLAMA_H, LLAMA_H)]
    if not quick:
        shapes = [
            (1, LLAMA_H, LLAMA_H),
            (1, LLAMA_H, LLAMA_I),
            (1, LLAMA_I, LLAMA_H),
            (4, LLAMA_H, LLAMA_H),
            (8, LLAMA_H, LLAMA_H),
            (16, LLAMA_H, LLAMA_H),
            (32, LLAMA_H, LLAMA_H),
            (128, LLAMA_H, LLAMA_H),
        ]
    for m, n, k in shapes:
        x = torch.randn(m, k, device="cuda", dtype=dtype) * (k**-0.5)
        weight = torch.randn(n, k, device="cuda", dtype=dtype) * (k**-0.5)
        # M > 16 cannot run the hand-written kernel. The dispatch row is the measurement.
        kinds = ["dot", "simt"] if m <= 16 else []
        eager_us = time_us(lambda x=x, weight=weight: F.linear(x, weight), iters=50 if m <= 16 else 20)
        mv_us = None
        if m == 1:
            try:
                mv_us = time_us(lambda weight=weight, x=x: torch.mv(weight, x.reshape(-1)), iters=50)
            except RuntimeError as exc:
                mv_us = None
                print("torch.mv failed", exc)
        nbytes = _linear_nbytes(m, n, k, dtype)
        if m > 1:
            dispatch_us = time_us(lambda x=x, weight=weight: linear_decode(x, weight), iters=30, warmup=5, rounds=3)
            rows.append(
                {
                    "name": f"linear M={m} N={n} K={k} dispatch {dtype}",
                    "us": dispatch_us,
                    "eager_us": eager_us,
                    "mv_us": mv_us,
                    "speedup_vs_eager": _speedup(eager_us, dispatch_us),
                    "bytes": nbytes,
                    "note": "default dispatcher, cuBLAS above M=1",
                }
            )
            print(f"linear M={m} N={n} K={k} dispatch {dispatch_us:.2f} us linear {eager_us:.2f}")
        for kind in kinds:
            def run(x=x, weight=weight, kind=kind):
                return linear_decode(x, weight, kind=kind)

            ours_us = time_us(run, iters=50 if m <= 16 else 20)
            ref = linear_reference(x, weight)
            ours = linear_decode(x, weight, kind=kind)
            cublas = F.linear(x, weight)
            seconds = ours_us * 1e-6
            useful_flops = 2.0 * m * n * k
            issued_flops = 2.0 * (16 if kind == "dot" and m <= 16 else m) * n * k
            row = {
                "name": f"linear M={m} N={n} K={k} {kind} {dtype}",
                "us": ours_us,
                "eager_us": eager_us,
                "mv_us": mv_us,
                "speedup_vs_eager": _speedup(eager_us, ours_us),
                "speedup_vs_mv": _speedup(mv_us, ours_us) if mv_us else None,
                "gbs": nbytes / seconds / 1e9,
                "pct_copy_roof": (nbytes / seconds / 1e9) / roof_gbs * 100.0 if roof_gbs else None,
                "useful_tflops": useful_flops / seconds / 1e12,
                "issued_tflops": issued_flops / seconds / 1e12,
                "issued_pct_cublas_peak": (issued_flops / seconds / 1e12) / cublas_tflops * 100.0 if cublas_tflops else None,
                "max_abs_vs_fp32": (ours.float() - ref.float()).abs().max().item(),
                "max_abs_vs_cublas": (ours.float() - cublas.float()).abs().max().item(),
                "bytes": nbytes,
            }
            rows.append(row)
            print(
                row["name"],
                f"{ours_us:.2f} us",
                f"linear {eager_us:.2f}",
                f"mv {mv_us:.2f}" if mv_us else "mv n/a",
                f"abs {row['max_abs_vs_fp32']:.3e}",
            )
    return rows


def _mlp_factory(dtype, m, kind):
    hidden = torch.randn(m, LLAMA_H, device="cuda", dtype=dtype)
    norm_w = torch.ones(LLAMA_H, device="cuda", dtype=dtype)
    w_gu = torch.randn(2 * LLAMA_I, LLAMA_H, device="cuda", dtype=dtype) * (LLAMA_H**-0.5)
    w_down = torch.randn(LLAMA_H, LLAMA_I, device="cuda", dtype=dtype) * (LLAMA_I**-0.5)
    i = LLAMA_I

    def eager(hidden=hidden, norm_w=norm_w, w_gu=w_gu, w_down=w_down, i=i):
        h = F.rms_norm(hidden, (LLAMA_H,), norm_w, 1e-6)
        gate, up = F.linear(h, w_gu).split(i, dim=-1)
        return F.linear(F.silu(gate) * up, w_down) + hidden

    def ours(hidden=hidden, norm_w=norm_w, w_gu=w_gu, w_down=w_down, kind=kind):
        return post_attn_mlp(hidden, norm_w, w_gu, w_down, kind=kind)

    return hidden, eager, ours


def bench_mlp(dtype: torch.dtype, quick: bool, compile_modes: bool) -> list[dict]:
    rows = []
    batches = [1] if quick else [1, 8]
    for m in batches:
        # None is the dispatcher: hand-written GEMV at M=1, cuBLAS above that.
        kinds = [None, "simt"] if m == 1 else [None]
        for kind in kinds:
            hidden, eager, ours = _mlp_factory(dtype, m, kind)
            ours_us = time_us(ours, iters=30, warmup=10)
            eager_us = time_us(eager, iters=30, warmup=10)
            with torch.no_grad():
                delta = (ours() - eager()).abs().max().item()
            label = "default" if kind is None else kind
            row = {
                "name": f"post_attn_mlp M={m} {label} {dtype}",
                "us": ours_us,
                "eager_us": eager_us,
                "speedup_vs_eager": _speedup(eager_us, ours_us),
                "max_abs_vs_eager": delta,
            }
            if compile_modes and kind is None and dtype == torch.float16:
                for mode in ("default", "reduce-overhead"):
                    try:
                        compiled = torch.compile(eager, mode=mode, dynamic=False)
                        # Compilation happens on the first calls. Keep it out of the timed loop.
                        compiled_us = time_us(compiled, iters=30, warmup=5)
                        row[f"compile_{mode}_us"] = compiled_us
                        row[f"speedup_vs_compile_{mode}"] = _speedup(compiled_us, ours_us)
                    except Exception as exc:
                        row[f"compile_{mode}_error"] = f"{type(exc).__name__}: {exc}"
                        print("compile failed", mode, exc)
            graph_us, graph_err = _graph_time(ours)
            row["cuda_graph_us"] = graph_us
            if graph_err:
                row["cuda_graph_error"] = graph_err
            rows.append(row)
            print(
                row["name"],
                f"{ours_us:.2f} us",
                f"eager {eager_us:.2f}",
                f"graph {graph_us:.2f}" if graph_us else f"graph failed: {graph_err}",
            )
    return rows


def _graph_time(fn):
    for _ in range(3):
        fn()
    _sync()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            fn()
    except Exception as exc:
        return None, f"{type(exc).__name__}: {exc}"
    _sync()
    start = torch.cuda.Event(True)
    end = torch.cuda.Event(True)
    samples = []
    for _ in range(5):
        start.record()
        for _ in range(30):
            graph.replay()
        end.record()
        _sync()
        samples.append(start.elapsed_time(end) / 30.0 * 1000.0)
    return statistics.median(samples), None


def tune(dtype: torch.dtype) -> None:
    configs = {
        "dot": [
            (64, 64, 4, 3),
            (64, 128, 4, 3),
            (128, 64, 4, 4),
            (128, 128, 8, 3),
            (32, 128, 4, 3),
            (64, 256, 4, 3),
        ],
        "simt": [
            (32, 64, 4, 2),
            (64, 64, 4, 3),
            (64, 128, 4, 3),
            (128, 64, 4, 3),
            (32, 128, 2, 3),
            (128, 128, 8, 3),
        ],
    }
    m, n, k = 1, LLAMA_H, LLAMA_H
    x = torch.randn(m, k, device="cuda", dtype=dtype)
    weight = torch.randn(n, k, device="cuda", dtype=dtype)
    y = torch.empty(m, n, device="cuda", dtype=dtype)
    previous = os.environ.get("FUSED_DECODE_LINEAR")
    try:
        for kind, choices in configs.items():
            best = None
            for block_n, block_k, warps, stages in choices:
                os.environ["FUSED_DECODE_LINEAR"] = f"{block_n},{block_k},{warps},{stages}"
                try:
                    us = time_us(lambda: linear_out(x, weight, y, kind=kind), iters=30, warmup=3, rounds=3)
                except Exception as exc:
                    print(kind, (block_n, block_k, warps, stages), "FAIL", exc)
                    continue
                print(f"{kind} {(block_n, block_k, warps, stages)} {us:.2f} us")
                if best is None or us < best[0]:
                    best = (us, (block_n, block_k, warps, stages))
            print("BEST", kind, best)
    finally:
        if previous is None:
            os.environ.pop("FUSED_DECODE_LINEAR", None)
        else:
            os.environ["FUSED_DECODE_LINEAR"] = previous


def _print_rows(rows: list[dict]) -> None:
    print("\nname | us | eager_us | speedup | gbs | pct_roof")
    for row in rows:
        speed = row.get("speedup_vs_eager")
        gbs = row.get("gbs")
        pct = row.get("pct_copy_roof")
        speed_s = f"{speed:.2f}x" if speed else ""
        gbs_s = f"{gbs:.1f}" if gbs is not None else ""
        pct_s = f"{pct:.1f}" if pct is not None else ""
        eager = row.get("eager_us")
        eager_s = f"{eager:.2f}" if eager is not None else ""
        print(f"{row['name']} | {row['us']:.2f} | {eager_s} | {speed_s} | {gbs_s} | {pct_s}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark fused-decode against PyTorch")
    parser.add_argument("--out", type=Path, default=Path("results/latest.json"))
    parser.add_argument("--compile", action="store_true", help="also time torch.compile on the MLP block")
    parser.add_argument("--quick", action="store_true", help="fp16 headline shapes only")
    parser.add_argument("--tune", action="store_true", help="search linear tile sizes and exit")
    parser.add_argument("--dtype", choices=("float16", "bfloat16", "both"), default="both")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    dtypes = [torch.float16] if args.quick or args.dtype == "float16" else []
    if args.dtype == "bfloat16":
        dtypes = [torch.bfloat16]
    elif args.dtype == "both" and not args.quick:
        dtypes = [torch.float16, torch.bfloat16]
    if args.quick:
        dtypes = [torch.float16]

    if args.tune:
        for dtype in dtypes:
            print("tuning", dtype)
            tune(dtype)
        return

    payload = {
        "gpu": _gpu_info(),
        "torch": torch.__version__,
        "triton": __import__("triton").__version__,
        "allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "allow_fp16_reduced_precision_reduction": bool(
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
        ),
        "clocks_under_load": _sample_clocks(),
        "roofline": [],
        "rows": [],
    }
    print(json.dumps({k: payload[k] for k in ("gpu", "torch", "triton", "clocks_under_load")}, indent=2))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for dtype in dtypes:
        roof = bench_roofline(dtype)
        payload["roofline"].append(roof)
        print(
            f"roof {dtype}: copy {roof['copy_gbs']:.1f} GB/s, triad {roof['triad_gbs']:.1f} GB/s, "
            f"cublas {roof['cublas_tflops']:.2f} TFLOP/s"
        )
        payload["rows"].extend(bench_epilogue(dtype, roof["copy_gbs"], args.quick, args.compile))
        payload["rows"].extend(bench_linear(dtype, roof["copy_gbs"], roof["cublas_tflops"], args.quick))
        payload["rows"].extend(bench_mlp(dtype, args.quick, args.compile))
        args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _print_rows(payload["rows"])
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
