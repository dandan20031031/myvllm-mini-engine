"""Alternating same-runner FP16 single-request eager/CUDA Graph benchmark.

No profiler, model hooks, or per-token instrumentation are installed. First
Graph calls and warmup are reported separately from paired warmed requests.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_engine.model import NativeQwen3Runner
from mini_engine.runtime import configure_local_cache
from benchmarks.common import gpu_snapshot, percentiles, sha256_json, source_metadata


def graph_status(runner):
    """Use an optional public status API without inspecting or patching kernels."""
    status = getattr(runner, "chat_graph_status", None)
    return status() if callable(status) else None


def execute_request(runner, prompt, budget, *, use_cuda_graph):
    started = time.perf_counter()
    result = runner.generate_incremental(prompt, budget, temperature=0.0, use_cuda_graph=use_cuda_graph)
    outer_ms = (time.perf_counter() - started) * 1000.0
    return {
        "mode": "cuda_graph" if use_cuda_graph else "eager",
        "token_ids": result.token_ids, "output_tokens": len(result.token_ids),
        "output_token_sha256": sha256_json(result.token_ids),
        "finish_reason": result.finish_reason,
        "ttft_ms": result.ttft_ms, "tpot_ms": result.tpot_ms,
        "total_latency_ms": result.total_latency_ms, "outer_call_wall_ms": outer_ms,
    }


def requests_match(left, right):
    return left["token_ids"] == right["token_ids"] and left["finish_reason"] == right["finish_reason"]


def summarize_case(pairs, *, comparison_allowed=True):
    modes = {}
    for mode in ("eager", "cuda_graph"):
        rows = [pair[mode] for pair in pairs]
        tokens = sum(row["output_tokens"] for row in rows)
        latency_ms = sum(row["total_latency_ms"] for row in rows)
        outer_ms = sum(row["outer_call_wall_ms"] for row in rows)
        modes[mode] = {
            "requests": len(rows), "actual_output_tokens": tokens,
            "ttft_ms": percentiles(row["ttft_ms"] for row in rows),
            "tpot_ms": percentiles(row["tpot_ms"] if row["output_tokens"] > 1 else None for row in rows),
            "total_latency_ms": percentiles(row["total_latency_ms"] for row in rows),
            "outer_call_wall_ms": percentiles(row["outer_call_wall_ms"] for row in rows),
            "sum_request_latency_ms": latency_ms, "sum_outer_call_wall_ms": outer_ms,
            "total_tokens_per_second": tokens * 1000.0 / latency_ms if latency_ms > 0 else None,
            "outer_call_tokens_per_second": tokens * 1000.0 / outer_ms if outer_ms > 0 else None,
        }
    output_match = all(pair["outputs_match"] for pair in pairs)
    valid = bool(pairs) and output_match and comparison_allowed
    return {
        "outputs_match": output_match, "comparison_allowed": valid, "modes": modes,
        "graph_over_eager_throughput_ratio": modes["cuda_graph"]["total_tokens_per_second"] / modes["eager"]["total_tokens_per_second"] if valid and modes["eager"]["total_tokens_per_second"] else None,
        "note": "Total TPS is sum(actual emitted tokens)/sum(request latency), not the median of per-request TPS. Outer-call TPS additionally includes tokenization/text decoding and wrapper overhead. GPU snapshots and metadata collection are excluded from both.",
    }


def capture_count(status):
    if not isinstance(status, dict):
        return None
    count = status.get("capture_count")
    return count if type(count) is int else None


def run_case(runner, prompt, budget, args):
    before_gpu = gpu_snapshot()
    before_setup_graph = graph_status(runner)
    # Both setup calls are excluded: the eager call warms common kernels before
    # the first Graph request encounters any missing context/table buckets.
    first_eager = execute_request(runner, prompt, budget, use_cuda_graph=False)
    first_graph = execute_request(runner, prompt, budget, use_cuda_graph=True)
    after_setup_graph = graph_status(runner)
    warmup = []
    for repetition in range(args.warmup):
        order = ("eager", "cuda_graph") if repetition % 2 == 0 else ("cuda_graph", "eager")
        rows = {mode: execute_request(runner, prompt, budget, use_cuda_graph=mode == "cuda_graph") for mode in order}
        warmup.append({"execution_order": list(order), "outputs_match": requests_match(rows["eager"], rows["cuda_graph"]), **rows})
    before_measured_graph = graph_status(runner)
    pairs = []
    for repetition in range(args.repetitions):
        order = ("eager", "cuda_graph") if repetition % 2 == 0 else ("cuda_graph", "eager")
        started = time.perf_counter()
        rows = {mode: execute_request(runner, prompt, budget, use_cuda_graph=mode == "cuda_graph") for mode in order}
        pair_ms = (time.perf_counter() - started) * 1000.0
        pairs.append({"repetition": repetition, "execution_order": list(order), "outputs_match": requests_match(rows["eager"], rows["cuda_graph"]), "pair_wall_ms": pair_ms, **rows})
    after_measured_graph = graph_status(runner)
    old_count, new_count = capture_count(before_measured_graph), capture_count(after_measured_graph)
    measured_capture = new_count != old_count if old_count is not None and new_count is not None else None
    all_checks_match = requests_match(first_eager, first_graph) and all(row["outputs_match"] for row in warmup + pairs)
    return {
        "prompt": prompt, "max_new_tokens": budget,
        "first_calls": {"eager": first_eager, "cuda_graph": first_graph, "outputs_match": requests_match(first_eager, first_graph), "pure_capture_ms": None, "note": "The first Graph request in this case can include capture plus prefill/decode; it is not a capture-only timer. Already captured buckets may be reused across cases."},
        "graph_status_before_setup": before_setup_graph, "graph_status_after_setup": after_setup_graph,
        "warmup": warmup, "pairs": pairs,
        "graph_status_before_measured": before_measured_graph, "graph_status_after_measured": after_measured_graph,
        "capture_during_measured_requests": measured_capture,
        "summary": summarize_case(pairs, comparison_allowed=all_checks_match and measured_capture is not True),
        "gpu_before_case": before_gpu, "gpu_after_case": gpu_snapshot(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--max-new-tokens", type=int, choices=[64, 128], action="append", dest="budgets")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--seed", type=int, default=41)
    parser.add_argument("--prefill-backend", choices=["auto", "math", "cuda", "dandan"], default="cuda")
    parser.add_argument("--linear-backend", choices=["cuda", "dandan", "dandan_cuda"], default="cuda")
    parser.add_argument("--sampling-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--fuse-projections", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "results" / "chat_graph.json")
    args = parser.parse_args()
    if not Path(args.model).is_dir():
        parser.error("a complete local model directory is required")
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        parser.error("real FP16 CUDA benchmarking requires an available CUDA device")
    if args.warmup < 0 or args.repetitions <= 0 or min(args.num_blocks, args.block_size) <= 0:
        parser.error("warmup >= 0 and positive repetitions/block counts are required")
    args.budgets = args.budgets or [64, 128]
    args.prompts = args.prompts or ["Explain KV Cache in one paragraph.", "用一小段话解释 continuous batching。"]
    configure_local_cache()
    torch.manual_seed(args.seed)
    source_start = source_metadata(extra_files=(__file__,))
    runner = NativeQwen3Runner.from_pretrained(
        args.model, device=args.device, dtype=torch.float16,
        prefill_backend=args.prefill_backend, decode_backend="cuda",
        elementwise_backend="cuda", data_backend="cuda", linear_backend=args.linear_backend,
        sampling_backend=args.sampling_backend, fuse_projections=args.fuse_projections,
        num_blocks=args.num_blocks, block_size=args.block_size, local_files_only=True,
    )
    backend_start = runner.model.backend_report()
    previous_eos = runner.eos_token_ids
    try:
        if args.ignore_eos:
            runner.eos_token_ids = []
        cases = [run_case(runner, prompt, budget, args) for prompt in args.prompts for budget in args.budgets]
    finally:
        runner.eos_token_ids = previous_eos
        runner.close()
    backend_end = runner.model.backend_report()
    source_end = source_metadata(extra_files=(__file__,))
    stable_files = source_start["source_tree_sha256"] == source_end["source_tree_sha256"] and backend_start == backend_end
    aligned = all(case["summary"]["comparison_allowed"] for case in cases)
    if not stable_files:
        for case in cases:
            case["summary"]["comparison_allowed"] = False
            case["summary"]["graph_over_eager_throughput_ratio"] = None
    payload = {
        "status": "passed_output_alignment" if stable_files and aligned else "comparison_not_validated",
        "model": Path(args.model).name, "device": args.device,
        "dtype": "fp16", "pytorch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "client_os": platform.platform(), "seed": args.seed,
        "configuration": {"budgets": args.budgets, "warmup_pairs": args.warmup, "repetitions": args.repetitions, "ignore_eos": args.ignore_eos, "projection_fusion": args.fuse_projections, "block_size": args.block_size, "num_blocks": args.num_blocks, "sampling": "greedy", "sampling_backend": args.sampling_backend},
        "backend_report_start": backend_start, "backend_report_end": backend_end,
        "source_start": source_start, "source_end": source_end,
        "source_and_backend_files_unchanged": stable_files,
        "cases": cases,
        "timing_scope": "Same loaded runner and weights; generate_incremental(use_cuda_graph=False/True) changes only the per-call decode path. Runner latency spans prefill through last emitted token, excluding tokenization/text decode; outer call wall time includes those. First calls, warmup, nvidia-smi, file hashes and git queries are excluded from warmed summaries. No profiler or model hooks are active.",
        "limitations": [
            "Single-request local generation, not service latency or continuous-batching throughput.",
            "Per-request TPOT is averaged over generated intervals; its P50/P95 describe requests, not the distribution of individual token intervals.",
            "GPU snapshots and process lists do not establish exclusive hardware use, especially under Windows WDDM.",
            "A first Graph request is capture-inclusive; pure capture time is unavailable unless separately provided by the runner status API.",
            "Identical file fingerprints and token outputs are necessary checks, not independent verification of an uncontended GPU.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    # Native Windows pipes may use GBK. Preserve UTF-8 in the file while
    # keeping console output representable even for diagnostic replacement chars.
    print(json.dumps(payload, ensure_ascii=True, indent=2))
    raise SystemExit(0 if stable_files and aligned else 1)


if __name__ == "__main__":
    main()
