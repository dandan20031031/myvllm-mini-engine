"""Independent-process PagedAttention pilot; never an engine throughput test.

Run this file with each provider's existing interpreter. Compare input hashes
before using either environment's timings. Mini/Triton imports are lazy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from benchmarks.common import (gpu_snapshot, percentile, run_metadata_command,
                               sha256_json, source_metadata)


TENSOR_NAMES = ("query", "key_cache", "value_cache", "block_tables", "context_lens")


def make_case(batch, context, *, heads=16, kv_heads=8, head_dim=128,
              block_size=256, seed=20261026):
    """Build CPU tensors with a private generator; hash actual bytes, not seeds."""
    import torch

    if min(batch, context, heads, kv_heads, head_dim, block_size) <= 0 or heads % kv_heads:
        raise ValueError("positive dimensions and heads divisible by KV heads are required")
    width = math.ceil(context / block_size)
    blocks = batch * width + 7
    generator = torch.Generator(device="cpu").manual_seed(seed + batch * 1000003 + context)
    query = torch.randn(batch, heads, head_dim, generator=generator, dtype=torch.float32).half()
    key = torch.randn(blocks, block_size, kv_heads, head_dim, generator=generator,
                      dtype=torch.float32).half()
    value = torch.randn(key.shape, generator=generator, dtype=torch.float32).half()
    table = torch.randperm(blocks, generator=generator)[:batch * width].reshape(batch, width)
    lengths = torch.full((batch,), context, dtype=torch.int64)
    tensors = dict(zip(TENSOR_NAMES, (query, key, value, table, lengths)))
    manifest = {name: {"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                       "sha256": hashlib.sha256(memoryview(tensor.view(torch.uint8).numpy())).hexdigest()}
                for name, tensor in tensors.items()}
    shape = {"batch": batch, "context": context, "heads": heads, "kv_heads": kv_heads,
             "head_dim": head_dim, "block_size": block_size, "seed": seed}
    return tensors, {"shape": shape, "tensors": manifest,
                     "case_sha256": sha256_json({"shape": shape, "tensors": manifest})}


def fp32_oracle(tensors, block_size):
    """Paged GQA oracle with FP32 QK, softmax and PV, outside all timings."""
    import torch

    q, key, value, tables, lengths = (tensors[name] for name in TENSOR_NAMES)
    outputs = []
    for row, length in enumerate(lengths.cpu().tolist()):
        ids = tables[row, :math.ceil(length / block_size)]
        k = key[ids].reshape(-1, key.shape[2], key.shape[3])[:length].float()
        v = value[ids].reshape(-1, value.shape[2], value.shape[3])[:length].float()
        group = q.shape[1] // k.shape[1]
        scores = torch.einsum("hd,thd->ht", q[row].float(), k.repeat_interleave(group, dim=1))
        probability = (scores / math.sqrt(q.shape[-1])).softmax(-1)
        outputs.append(torch.einsum("ht,thd->hd", probability, v.repeat_interleave(group, dim=1)))
    return torch.stack(outputs)


def validate_output(actual, reference, *, atol=0.006, rtol=0.03):
    import torch

    if actual.shape != reference.shape:
        return {"passed": False, "reason": "shape mismatch", "atol": atol, "rtol": rtol}
    actual = actual.float()
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(reference).all())
    error = (actual - reference).abs()
    passed = finite and bool(torch.allclose(actual, reference, atol=atol, rtol=rtol))
    return {"passed": passed, "finite": finite, "atol": atol, "rtol": rtol,
            "max_absolute_error": error.max().item() if finite else None,
            "rms_error": error.square().mean().sqrt().item() if finite else None,
            "outside_tolerance_fraction": (error > atol + rtol * reference.abs()).float().mean().item()}


def _file_hashes(paths, base):
    return {str(path.relative_to(base)).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def provider_metadata(provider, module, mini_source=None, extension=None):
    result = {"module_file": str(Path(module.__file__).resolve())}
    if provider == "cuda":
        binary = Path(extension.__file__).resolve()
        result.update(binary_file=str(binary), binary_sha256=hashlib.sha256(binary.read_bytes()).hexdigest())
    else:
        import triton

        source = Path(mini_source).resolve()
        hashes = _file_hashes((source / "myvllm").rglob("*.py"), source)
        result.update(triton_version=triton.__version__, source_files_sha256=hashes,
                      source_tree_sha256=sha256_json(hashes),
                      mini_git=run_metadata_command(["git", "-C", str(source), "rev-parse", "HEAD"]))
        kernels = []
        # Inspection only: the provider's regular call compiled these kernels.
        for cache in module.paged_attention_decode_kernel.device_caches.values():
            for kernel in cache[0].values():
                kernels.append({"name": kernel.name, "triton_hash": kernel.hash,
                                "binary_sha256": hashlib.sha256(kernel.kernel).hexdigest()})
        result["compiled_kernels"] = sorted(kernels, key=lambda row: row["triton_hash"])
    return result


def load_provider(provider, mini_source=None):
    if provider == "cuda":
        module = importlib.import_module("mini_engine.kernels.paged")
        from mini_engine.kernels.cuda_extension import load_cuda_extension
        extension = load_cuda_extension()

        def call(tensors, block_size):
            return module.paged_attention(*(tensors[name] for name in TENSOR_NAMES), block_size,
                                          backend="cuda", validate=False)
        return call, lambda: provider_metadata(provider, module, extension=extension)
    if not mini_source:
        raise ValueError("--mini-source must point to the selected checkout's src directory")
    source = Path(mini_source).resolve()
    expected = source / "myvllm" / "layers" / "attention.py"
    if not expected.is_file():
        raise ValueError("--mini-source does not contain myvllm/layers/attention.py")
    sys.path.insert(0, str(source))
    module = importlib.import_module("myvllm.layers.attention")
    if Path(module.__file__).resolve() != expected:
        raise RuntimeError("Mini import resolved to a different checkout")

    def call(tensors, block_size):
        q = tensors["query"]
        return module.paged_attention_decode(*(tensors[name] for name in TENSOR_NAMES),
                                              1 / math.sqrt(q.shape[-1]), q.shape[1],
                                              tensors["key_cache"].shape[2], q.shape[-1], block_size)
    return call, lambda: provider_metadata(provider, module, source)


def measure(call, tensors, block_size, *, calls=50, repetitions=6, warmup=2):
    import torch

    for _ in range(warmup):
        call(tensors, block_size)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    # Compile/allocate before capture; current stream is used by both providers.
    with torch.cuda.graph(graph):
        for _ in range(calls):
            graph_output = call(tensors, block_size)
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    samples = {"graph_device_event_ms_per_call": [], "eager_wall_ms_per_call": []}
    last = graph_output
    for repetition in range(repetitions):
        order = ("graph", "eager") if repetition % 2 == 0 else ("eager", "graph")
        for mode in order:
            torch.cuda.synchronize()
            if mode == "graph":
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples["graph_device_event_ms_per_call"].append(start.elapsed_time(end) / calls)
            else:
                started = time.perf_counter()
                for _ in range(calls):
                    last = call(tensors, block_size)
                torch.cuda.synchronize()
                samples["eager_wall_ms_per_call"].append((time.perf_counter() - started) * 1000 / calls)
    summary = {name: {"count": len(values), "mean": statistics.mean(values),
                      "p50": percentile(values, .5), "p95": percentile(values, .95)}
               for name, values in samples.items()}
    return {"samples": samples, "summary": summary}, last, graph_output


def comparison_guard(reports):
    """Reject unmatched/failed reports; matching inputs do not equal matching engines."""
    issues = []
    if len(reports) < 2:
        issues.append("at least two runs are required")
    baseline = reports[0] if reports else {}
    baseline_cases = [row.get("input", {}).get("case_sha256") for row in baseline.get("cases", [])]
    base_gpus = [gpu.get("uuid") for gpu in baseline.get("gpu_before", {}).get("gpus", [])]
    for index, report in enumerate(reports):
        if report.get("status") != "passed" or not report.get("source_unchanged"):
            issues.append(f"run {index}: incomplete, failed or source changed")
        cases = report.get("cases", [])
        if not cases or not all(row.get("correctness", {}).get("passed") for row in cases):
            issues.append(f"run {index}: missing/pending correctness")
        if [row.get("input", {}).get("case_sha256") for row in cases] != baseline_cases:
            issues.append(f"run {index}: input tensor hashes differ")
        if report.get("measurement") != baseline.get("measurement"):
            issues.append(f"run {index}: measurement settings differ")
        gpus = [gpu.get("uuid") for gpu in report.get("gpu_before", {}).get("gpus", [])]
        after = [gpu.get("uuid") for gpu in report.get("gpu_after", {}).get("gpus", [])]
        if not gpus or not all(gpus) or gpus != base_gpus or after != gpus:
            issues.append(f"run {index}: same GPU identity not established")
    return {"component_comparison_allowed": not issues, "issues": issues,
            "same_environment": len({sha256_json(report.get("environment", {})) for report in reports}) == 1,
            "engine_throughput_comparison_allowed": False,
            "note": "Independent environments; no engine TPS, automatic speed ratio or isolation guarantee."}


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--provider", choices=("cuda", "mini"), required=True)
    result.add_argument("--mini-source", type=Path)
    result.add_argument("--batch", type=int, nargs="+", default=[1, 8, 32])
    result.add_argument("--context", type=int, nargs="+", default=[128, 512, 2048])
    result.add_argument("--seed", type=int, default=20261026)
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--repetitions", type=int, default=6)
    result.add_argument("--calls", type=int, default=50)
    result.add_argument("--output", type=Path, required=True)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    if len(args.batch) != len(args.context) or min(args.calls, args.repetitions) <= 0 or args.warmup < 1:
        raise SystemExit("paired batch/context values, positive calls/repetitions and warmup >= 1 required")
    from mini_engine.runtime import configure_local_cache
    configure_local_cache()
    os.environ["TRITON_CACHE_DIR"] = str(PROJECT_ROOT / ".cache" / "triton_compare")
    import torch

    report = {"status": "failed", "provider": args.provider, "cases": [],
              "environment": {"python": sys.version, "python_executable": sys.executable,
                              "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                              "nvcc": run_metadata_command(["nvcc", "--version"])},
              "measurement": {"calls_per_sample": args.calls, "repetitions": args.repetitions,
                              "warmup": args.warmup, "atol": .006, "rtol": .03,
                              "shape": "FP16 Hq16/Hkv8/D128/block256 random physical pages",
                              "graph": "50-op graph by default; device events divided by op count",
                              "eager": "Python callback launches plus final CUDA sync; wall divided by op count",
                              "excluded": "CPU generation/hash/H2D, oracle, validation, import/JIT, capture, warmup, metadata"},
              "gpu_before": gpu_snapshot(),
              "source_before": source_metadata(extra_files=(Path(__file__),))}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        report["environment"].update(gpu_name=torch.cuda.get_device_name(),
                                      capability=list(torch.cuda.get_device_capability()))
        call, metadata = load_provider(args.provider, args.mini_source)
        report["provider_before"] = metadata()
        with torch.inference_mode():
            for batch, context in zip(args.batch, args.context):
                cpu_tensors, manifest = make_case(batch, context, seed=args.seed)
                tensors = {name: tensor.cuda() for name, tensor in cpu_tensors.items()}
                del cpu_tensors
                oracle = fp32_oracle(tensors, 256)
                correctness = validate_output(call(tensors, 256), oracle)
                row = {"input": manifest, "correctness": correctness}
                report["cases"].append(row)
                if not correctness["passed"]:
                    raise RuntimeError(f"correctness failed before timing B={batch}, context={context}")
                timing, eager_output, graph_output = measure(call, tensors, 256, calls=args.calls,
                    repetitions=args.repetitions, warmup=args.warmup)
                row["timing"] = timing
                row["post_timing_correctness"] = {"eager": validate_output(eager_output, oracle),
                                                   "graph": validate_output(graph_output, oracle)}
                if not all(value["passed"] for value in row["post_timing_correctness"].values()):
                    row["correctness"]["passed"] = False
                    raise RuntimeError("post-timing output differs from oracle")
                print(json.dumps({"batch": batch, "context": context,
                                  "correctness": correctness, "summary": timing["summary"]}), flush=True)
                del tensors, oracle, eager_output, graph_output
        report["provider_after"] = metadata()
        report["source_after"] = source_metadata(extra_files=(Path(__file__),))
        report["source_unchanged"] = (
            report["source_before"]["source_tree_sha256"] == report["source_after"]["source_tree_sha256"]
            and report["provider_before"].get("source_tree_sha256") == report["provider_after"].get("source_tree_sha256")
            and report["provider_before"].get("binary_sha256") == report["provider_after"].get("binary_sha256"))
        if not report["source_unchanged"]:
            raise RuntimeError("provider/benchmark sources or binary changed during run")
        report["status"] = "passed"
    except Exception:
        report["error"] = traceback.format_exc()
    finally:
        report["gpu_after"] = gpu_snapshot()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    if report["status"] != "passed":
        print(report.get("error"), file=sys.stderr)
        return 1
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
