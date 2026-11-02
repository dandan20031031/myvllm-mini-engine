"""Same-checkpoint old/new decode-kernel validation and paired generation.

The route override is local to this offline experiment. Normal CLI/backend
configuration stays unchanged; each runner captures its own Graph.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from mini_engine.runtime import configure_local_cache
configure_local_cache()

import torch
from mini_engine.model import NativeQwen3Runner
from benchmarks.common import gpu_snapshot, percentiles, sha256_json, source_metadata
from benchmarks.benchmark_chat_graph import execute_request
from benchmarks.diagnose_ragged_prefill import compare_teacher_forced, teacher_forced_trace, clean_json
from mini_engine.engine.batch import PrefillBatch

ROUTES = ("cuda_vector128", "cuda")


@contextmanager
def kernel_route(backend):
    import mini_engine.model.native_qwen3 as model_module
    original = model_module.paged_attention
    def routed(*args, **kwargs):
        if kwargs.get("backend") == "cuda":
            kwargs["backend"] = backend
        return original(*args, **kwargs)
    model_module.paged_attention = routed
    try:
        yield
    finally:
        model_module.paged_attention = original


def forced_trace(runner, ids, tokens):
    manager, cache = runner.block_manager, runner.cache
    pages = manager.allocate("__decode_round__", len(ids) + len(tokens))
    try:
        batch = PrefillBatch.from_sequences([ids], [pages], [0], block_size=manager.block_size,
            num_blocks=manager.num_blocks, device=cache.device)
        case = {"batch": batch, "singles": [batch], "single_rows": [(len(ids)-1,)],
                "output": torch.empty(1, runner.model.config.vocab_size, device=cache.device, dtype=cache.dtype)}
        table = torch.tensor([pages], device=cache.device, dtype=torch.long)
        return teacher_forced_trace(runner, case, "sequential_selected_logits", [ids], [pages], table, [tokens])
    finally:
        torch.cuda.synchronize()
        manager.release("__decode_round__")


def summarize(rows):
    return {"tpot_ms": percentiles(row["tpot_ms"] for row in rows),
            "ttft_ms": percentiles(row["ttft_ms"] for row in rows),
            "latency_ms": percentiles(row["total_latency_ms"] for row in rows),
            "actual_tokens": sum(row["output_tokens"] for row in rows),
            "tokens_per_second": sum(row["output_tokens"] for row in rows) * 1000 /
                sum(row["total_latency_ms"] for row in rows)}


def checkpoint_fingerprint(path):
    files = sorted({*Path(path).glob("*.safetensors"), *Path(path).glob("*.json")})
    result = {}
    for file in files:
        with file.open("rb") as handle:
            result[file.name] = hashlib.file_digest(handle, "sha256").hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models/Qwen3-0.6B"))
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.warmup < 0 or min(args.repetitions, args.max_new_tokens) <= 0:
        parser.error("warmup >= 0 and positive repetitions/output budget required")
    torch.set_num_threads(1)
    sources = (__file__, sys.modules[compare_teacher_forced.__module__].__file__,
               sys.modules[execute_request.__module__].__file__,
               sys.modules["benchmarks.benchmark_ragged_prefill"].__file__)
    source_start = source_metadata(extra_files=sources)
    checkpoint_hashes = checkpoint_fingerprint(args.model)
    payload = {"kind": "paired_decode_kernel_generation", "configuration": vars(args) | {"output": str(args.output)},
               "source_start": source_start, "checkpoint_sha256": checkpoint_hashes,
               "gpu_before": gpu_snapshot(), "cases": [], "completed": False,
               "numerical_protocol": {"atol": 0.125, "rtol": 0.01, "nonfinite_allowed": False,
                   "checks": "teacher-forced logits/newest KV every step, full greedy token sequence, eager/Graph alignment"},
               "scope": "Single request. Warmed runner latency spans prefill through last token; excludes loading, tokenization/text decode, capture, validation and cleanup. No HTTP/continuous-batch throughput claim."}
    try:
        with ExitStack() as stack:
            runners = {}
            for route in ROUTES:
                runner = NativeQwen3Runner.from_pretrained(args.model, dtype=getattr(torch,
                    "float16" if args.dtype == "fp16" else "bfloat16"), device="cuda", prefill_backend="cuda",
                    decode_backend="cuda", data_backend="cuda", elementwise_backend="cuda", linear_backend="cuda",
                    sampling_backend="cuda", fuse_projections=True, local_files_only=True, num_blocks=64, block_size=16)
                stack.callback(runner.close)
                runner.eos_token_ids = []
                runners[route] = runner
            payload["actual_execution"] = {route: {"parameter_dtypes": sorted({str(p.dtype) for p in r.model.parameters()}),
                "kv_dtype": str(r.cache.dtype), "backends": r.model.backend_report()} for route, r in runners.items()}
            for prompt in ("Explain KV Cache in one paragraph.", "用一小段话解释 continuous batching。"):
                ids = runners[ROUTES[0]].tokenizer.apply_chat_template([{"role":"user", "content":prompt}],
                    tokenize=True, add_generation_prompt=True, enable_thinking=False)
                first, warm = {}, []
                for route in ROUTES:
                    with kernel_route(route):
                        first[route] = execute_request(runners[route], prompt, args.max_new_tokens, use_cuda_graph=True)
                tokens = first[ROUTES[0]]["token_ids"]
                traces, eager_checks = {}, {}
                for route in ROUTES:
                    with kernel_route(route):
                        traces[route] = forced_trace(runners[route], ids, tokens)
                        eager_checks[route] = execute_request(runners[route], prompt, args.max_new_tokens, use_cuda_graph=False)
                numeric = compare_teacher_forced(traces[ROUTES[1]], traces[ROUTES[0]], 0.125, 0.01)
                del traces
                output_checks = all(row["token_ids"] == tokens and row["output_tokens"] == args.max_new_tokens
                                    for row in list(first.values()) + list(eager_checks.values()))
                allowed = numeric["all_steps_strict_passed"] and numeric["first_argmax_difference"] is None and output_checks
                case = {"prompt": prompt, "prompt_token_sha256": sha256_json(ids), "teacher_forced": numeric,
                        "first_calls": first, "eager_checks": eager_checks, "output_checks_passed": output_checks,
                        "comparison_allowed": allowed, "pairs": []}
                payload["cases"].append(case)
                if not allowed:
                    continue  # Preserve failure; no performance ratio for an invalid candidate.
                before_capture = {route: runners[route].chat_graph_status()["capture_count"] for route in ROUTES}
                for iteration in range(args.warmup + args.repetitions):
                    order = ROUTES if iteration % 2 == 0 else ROUTES[::-1]
                    row = {"order": list(order)}
                    for route in order:
                        with kernel_route(route):
                            row[route] = execute_request(runners[route], prompt, args.max_new_tokens, use_cuda_graph=True)
                        if row[route]["token_ids"] != tokens:
                            case["comparison_allowed"] = False
                    (warm if iteration < args.warmup else case["pairs"]).append(row)
                case["warmup"] = warm
                case["summary"] = {route: summarize([row[route] for row in case["pairs"]]) for route in ROUTES}
                case["graph_status"] = {route: runners[route].chat_graph_status() for route in ROUTES}
                if any(case["graph_status"][route]["capture_count"] != before_capture[route] for route in ROUTES):
                    case["comparison_allowed"] = False
            payload["completed"] = True
    finally:
        payload["gpu_after"] = gpu_snapshot()
        payload["source_end"] = source_metadata(extra_files=sources)
        stable = payload["source_start"]["source_tree_sha256"] == payload["source_end"]["source_tree_sha256"]
        payload["source_unchanged"] = stable
        payload["checkpoint_unchanged"] = checkpoint_hashes == checkpoint_fingerprint(args.model)
        payload["comparison_allowed"] = stable and payload["checkpoint_unchanged"] and payload["completed"] and bool(payload["cases"]) and all(
            case["comparison_allowed"] for case in payload["cases"])
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(clean_json(payload), ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "comparison_allowed": payload["comparison_allowed"],
                      "summary": [case.get("summary") for case in payload["cases"]]}, ensure_ascii=True))
    return 0 if payload["comparison_allowed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
