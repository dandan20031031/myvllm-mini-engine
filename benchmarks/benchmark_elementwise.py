r"""Stage 07 microbenchmark: CUDA Graph timeline and eager wall-clock time.

PowerShell: . .\scripts\enter_project.ps1
python benchmarks\benchmark_elementwise.py --output benchmarks\results\elementwise.json
No model weights are downloaded. A graph is used to reduce host dispatch gaps;
its event time still includes graph node scheduling, not just pure kernel time.
"""

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Configure project-local caches before loading torch/initializing CUDA.
from mini_engine.kernels.elementwise import residual_add, residual_rms_norm, rms_norm, silu_and_mul
import torch

OPERATIONS = {
    "rms_norm": lambda x, r, w, backend: rms_norm(x, w, backend=backend),
    "residual_rms_norm": lambda x, r, w, backend: residual_rms_norm(x, r, w, backend=backend),
    "residual_add": lambda x, r, w, backend: residual_add(x, r, backend=backend),
    "silu_and_mul": lambda x, r, w, backend: silu_and_mul(x, r, backend=backend),
}


def positive(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


@torch.inference_mode()
def measure(fn, *, warmup, repetitions, inner):
    # Warm up on a side stream before capture, as recommended by PyTorch.
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(warmup):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(inner):
            output = fn()
    graph.replay()
    torch.cuda.synchronize()

    pairs = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
             for _ in range(repetitions)]
    for start, end in pairs:
        start.record()
        graph.replay()
        end.record()
    torch.cuda.synchronize()
    graph_us = [start.elapsed_time(end) * 1000 / inner for start, end in pairs]

    # Each wall sample includes Python dispatch, allocation and one final sync.
    # It is intentionally not described as device kernel latency.
    wall_us = []
    for _ in range(5):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(repetitions):
            output = fn()
        torch.cuda.synchronize()
        wall_us.append((time.perf_counter() - start) * 1e6 / repetitions)
    result = {
        "graph_device_us_median": statistics.median(graph_us),
        "graph_device_us_samples": graph_us,
        "eager_wall_us_median": statistics.median(wall_us),
        "eager_wall_us_samples": wall_us,
    }
    del output, graph
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=positive, nargs="+", default=[1, 128])
    parser.add_argument("--hidden", type=positive, nargs="+", default=[1024, 3072])
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=positive, default=5)
    parser.add_argument("--repetitions", type=positive, default=20)
    parser.add_argument("--graph-inner", type=positive, default=32)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "results" / "elementwise.json")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    from mini_engine.kernels.cuda_extension import load_cuda_extension
    load_cuda_extension()
    torch.manual_seed(47)
    dtype = torch.float16 if args.dtype == "fp16" else torch.bfloat16
    results = []
    for rows in args.rows:
        for hidden in args.hidden:
            x = torch.randn(rows, hidden, device="cuda", dtype=dtype)
            residual = torch.randn_like(x)
            weight = torch.randn(hidden, device="cuda", dtype=dtype)
            for name, operation in OPERATIONS.items():
                reference = operation(x, residual, weight, "torch")
                actual = operation(x, residual, weight, "cuda")
                torch.testing.assert_close(actual, reference, atol=0.004 if dtype == torch.bfloat16 else 0.001, rtol=0.01)
                row = {"op": name, "rows": rows, "hidden": hidden, "dtype": args.dtype}
                # Alternate measurement order to reduce systematic order bias.
                backends = ["torch", "cuda"] if len(results) % 2 == 0 else ["cuda", "torch"]
                for backend in backends:
                    row[backend] = measure(
                        lambda: operation(x, residual, weight, backend),
                        warmup=args.warmup, repetitions=args.repetitions, inner=args.graph_inner,
                    )
                row["graph_ratio_torch_over_cuda"] = row["torch"]["graph_device_us_median"] / row["cuda"]["graph_device_us_median"]
                results.append(row)
                print(f"{name:18} rows={rows:3} hidden={hidden:4} graph: "
                      f"torch={row['torch']['graph_device_us_median']:.3f} us, "
                      f"cuda={row['cuda']['graph_device_us_median']:.3f} us, "
                      f"ratio={row['graph_ratio_torch_over_cuda']:.2f}")
    # Metadata applies to this run; no fabricated throughput from token budgets.
    payload = {
        "gpu": torch.cuda.get_device_name(0), "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda, "dtype": args.dtype,
        "warmup": args.warmup, "repetitions": args.repetitions, "graph_inner": args.graph_inner,
        "timing": "CUDA Event around graph replay / graph_inner; eager wall includes Python and allocation.",
        "comparison": "project elementwise torch reference; not vLLM or end-to-end inference",
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
