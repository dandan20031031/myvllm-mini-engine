"""Paired scalar/vectorized CUDA Graph timing; never a service latency claim."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mini_engine.runtime import configure_local_cache
from mini_engine.kernels.paged import paged_attention
from mini_engine.kernels.cuda_extension import load_cuda_extension


def gpu_snapshot():
    try:
        return subprocess.check_output(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,power.draw",
                                        "--format=csv,noheader"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture(call, repeats):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(repeats):
            output = call()
    return graph, output


def measure_pair(case, repetitions, graph_repeats):
    graphs = {}
    for backend in ("cuda_baseline", "cuda"):
        graphs[backend] = capture(lambda b=backend: paged_attention(*case, backend=b, validate=False), graph_repeats)
    samples = {name: [] for name in graphs}
    for iteration in range(repetitions):
        # Alternate order, avoiding systematic warm-clock advantage.
        names = list(graphs) if iteration % 2 == 0 else list(reversed(graphs))
        for name in names:
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            graphs[name][0].replay()
            end.record()
            end.synchronize()
            samples[name].append(begin.elapsed_time(end) / graph_repeats)
    return {name: {"samples_ms": values, "median_ms": statistics.median(values),
                   "mean_ms": statistics.mean(values)} for name, values in samples.items()}


def main():
    configure_local_cache()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[16, 32, 64, 128, 256])
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 8, 32])
    parser.add_argument("--contexts", nargs="+", type=int, default=[128, 512, 2048])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--graph-repeats", type=int, default=20)
    parser.add_argument("--output", default="benchmarks/results/paged_sweep.json")
    args = parser.parse_args()
    if (len(args.batches) != len(args.contexts) or not args.kv_heads or args.heads % args.kv_heads
            or min([*args.block_sizes, *args.batches, *args.contexts, args.repetitions, args.graph_repeats]) <= 0):
        parser.error("positive paired shapes and divisible GQA heads are required")
    extension = load_cuda_extension()
    before = gpu_snapshot()
    torch.manual_seed(26)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    rows = []
    for block in args.block_sizes:
        for batch, context in zip(args.batches, args.contexts):
            width = (context + block - 1) // block
            k = torch.randn(batch * width + 3, block, args.kv_heads, 128, device="cuda", dtype=dtype)
            v = torch.randn_like(k)
            tables = torch.randperm(k.shape[0], device="cuda")[:batch * width].reshape(batch, width)
            q = torch.randn(batch, args.heads, 128, device="cuda", dtype=dtype)
            lengths = torch.full((batch,), context, dtype=torch.long, device="cuda")
            case = (q, k, v, tables, lengths, block)
            expected = paged_attention(*case, backend="torch")
            actual = paged_attention(*case, backend="cuda")
            torch.testing.assert_close(actual, expected, atol=0.006 if dtype == torch.bfloat16 else 0.001, rtol=0.03)
            timing = measure_pair(case, args.repetitions, args.graph_repeats)
            row = {"batch": batch, "context": context, "block_size": block, "timing": timing,
                   "max_abs_error": (actual - expected).abs().max().item(),
                   "paired_kernel_speedup": timing["cuda_baseline"]["median_ms"] / timing["cuda"]["median_ms"]}
            rows.append(row)
            print(json.dumps({key: value for key, value in row.items() if key != "timing"}), flush=True)
    payload = {"scope": "CUDA Graph replay per operator; excludes Python dispatch and service",
               "external_load_not_controlled": True, "gpu_before": before, "gpu_after": gpu_snapshot(),
               "os": platform.platform(), "torch": torch.__version__, "cuda": torch.version.cuda,
               "extension_sha256": hashlib.sha256(Path(extension.__file__).read_bytes()).hexdigest(),
               "configuration": vars(args), "rows": rows}
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
