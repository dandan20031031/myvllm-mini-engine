"""Benchmark torch fallback versus the Windows CUDA paged decode kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_engine.kernels.paged import paged_attention
from mini_engine.runtime import configure_local_cache


def make_case(batch, heads, kv_heads, head_dim, context, block_size, dtype):
    width = (context + block_size - 1) // block_size
    blocks = batch * width + 3
    key = torch.randn(blocks, block_size, kv_heads, head_dim, device="cuda", dtype=dtype)
    value = torch.randn_like(key)
    table = torch.arange(batch * width, device="cuda", dtype=torch.long).view(batch, width)
    query = torch.randn(batch, heads, head_dim, device="cuda", dtype=dtype)
    lengths = torch.full((batch,), context, device="cuda", dtype=torch.long)
    return query, key, value, table, lengths, block_size


def time_backend(args, case, backend):
    for _ in range(args.warmup):
        paged_attention(*case, backend=backend, validate=False)
    torch.cuda.synchronize()
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        paged_attention(*case, backend=backend, validate=False)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {
        "backend": backend,
        "latencies_ms": times,
        "mean_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
        "peak_memory_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
    }


def main():
    configure_local_cache()
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--context", type=int, nargs="+", default=[128, 1024, 4096])
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=50)
    parser.add_argument("--output", default=str(PROJECT_ROOT / "benchmarks" / "results" / "paged_attention.json"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    if len(args.batch) != len(args.context):
        raise SystemExit("--batch and --context must have the same number of values")
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    from mini_engine.kernels.cuda_extension import load_cuda_extension
    load_cuda_extension(verbose=False)
    all_results = []
    for batch, context in zip(args.batch, args.context):
        case = make_case(batch, args.heads, args.kv_heads, args.head_dim, context, args.block_size, dtype)
        reference = paged_attention(*case, backend="torch", validate=False)
        cuda = paged_attention(*case, backend="cuda", validate=False)
        tolerance = 0.006 if dtype == torch.bfloat16 else 0.001
        torch.testing.assert_close(cuda, reference, atol=tolerance, rtol=0.03)
        row = {
            "batch": batch, "context": context, "heads": args.heads,
            "kv_heads": args.kv_heads, "head_dim": args.head_dim,
            "block_size": args.block_size, "dtype": args.dtype,
            "torch": time_backend(args, case, "torch"),
            "cuda": time_backend(args, case, "cuda"),
        }
        row["speedup"] = row["torch"]["mean_ms"] / row["cuda"]["mean_ms"]
        all_results.append(row)
        print(row)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"wrote {destination}")


if __name__ == "__main__":
    main()
