"""Compare prebuilt B1/full-logit, B1/selected-logit and ragged prefill forwards.

This is a cold-prefill forward benchmark, not generation or HTTP throughput.
All modes use the same loaded model, kernel backends, KV pool and input tokens.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
import time
import traceback
import uuid

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mini_engine.runtime import configure_local_cache

configure_local_cache()

import torch

from mini_engine.model import NativeQwen3Runner
from benchmarks.benchmark_chat_graph import gpu_snapshot, percentiles, sha256_json, source_metadata


MODES = ("sequential_full_logits", "sequential_selected_logits", "ragged_selected_logits")


class PrefillValidationError(AssertionError):
    def __init__(self, details):
        super().__init__("prefill numerical validation failed: " + "; ".join(details["failures"]))
        self.details = details


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def source_fingerprint():
    report = source_metadata()
    files = dict(report["source_files_sha256"])
    files[Path(__file__).relative_to(ROOT).as_posix()] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    report["source_files_sha256"] = files
    report["source_tree_sha256"] = sha256_json(files)
    return report


def ragged_lengths(batch_size, maximum):
    if batch_size <= 0 or maximum <= 0:
        raise ValueError("batch and prompt lengths must be positive")
    return [max(1, maximum * (4 - index % 4) // 4) for index in range(batch_size)]


def prompt_tokens(runner, lengths, text):
    source = runner.tokenizer.encode(text, add_special_tokens=False)
    if not source:
        raise ValueError("benchmark prompt tokenized to an empty sequence")
    # Tokenization and exact-length construction are outside all timers. There
    # is no PrefixCache or scheduler in these cold-forward measurements.
    return [(source * ((length + len(source) - 1) // len(source)))[:length] for length in lengths]


@contextmanager
def prepared_case(runner, sequences):
    from mini_engine.engine.batch import PrefillBatch

    if not sequences or any(not ids for ids in sequences):
        raise ValueError("nonempty token sequences are required")
    manager, cache = runner.block_manager, runner.cache
    owners, pages = [], []
    try:
        tag = uuid.uuid4().hex
        for index, ids in enumerate(sequences):
            owner = f"__prefill_benchmark__:{tag}:{index}"
            pages.append(manager.allocate(owner, len(ids)))
            owners.append(owner)
        common = {"block_size": manager.block_size, "num_blocks": manager.num_blocks, "device": cache.device}
        packed = PrefillBatch.from_sequences(sequences, pages, [0] * len(pages), **common)
        singles = [PrefillBatch.from_sequences([ids], [table], [0], **common)
                   for ids, table in zip(sequences, pages)]
        # The public forward validates indices on the host, then creates its
        # device index tensor. Passing a GPU tensor would add an avoidable D2H
        # read to every measured call. Only host selection planning is excluded.
        last_rows = tuple(offset - 1 for offset in packed.query_offsets[1:])
        single_rows = [(len(ids) - 1,) for ids in sequences]
        output = torch.empty((len(sequences), runner.model.config.vocab_size), device=cache.device, dtype=cache.dtype)
        yield {"batch": packed, "singles": singles, "last_rows": last_rows,
               "single_rows": single_rows, "pages": pages, "output": output,
               "total_tokens": sum(map(len, sequences)), "lengths": list(map(len, sequences))}
    finally:
        # Queued KV writes must finish before returning physical pages.
        synchronize(cache.device)
        for owner in owners:
            manager.release(owner)


@torch.inference_mode()
def execute_mode(runner, case, mode):
    model, cache, output = runner.model, runner.cache, case["output"]
    if mode == "ragged_selected_logits":
        rows = model.forward_prefill_batch(case["batch"], cache, logits_rows=case["last_rows"])
        output.copy_(rows)
    elif mode in MODES[:2]:
        for index, batch in enumerate(case["singles"]):
            selected = None if mode == "sequential_full_logits" else case["single_rows"][index]
            rows = model.forward_prefill_batch(batch, cache, logits_rows=selected)
            output[index:index + 1].copy_(rows[-1:])
            # Do not retain slices backed by all-token vocabulary logits.
            del rows
    else:
        raise ValueError(f"unknown benchmark mode: {mode}")
    return output


def cache_snapshot(cache, slots):
    shape = cache.shape
    flat_shape = (shape.num_layers, shape.num_blocks * shape.block_size, shape.num_kv_heads, shape.head_dim)
    return tuple(tensor.reshape(flat_shape).index_select(1, slots).detach().cpu()
                 for tensor in (cache.key, cache.value))


def correctness_checks(runner, case, *, atol, rtol, check_kv=True):
    logits, caches = {}, {}
    for mode in MODES:
        values = execute_mode(runner, case, mode)
        synchronize(runner.cache.device)
        logits[mode] = values.detach().float().cpu().clone()
        if not bool(torch.isfinite(logits[mode]).all()):
            raise AssertionError(f"{mode} produced nonfinite logits")
        if check_kv:
            caches[mode] = cache_snapshot(runner.cache, case["batch"].slot_mapping)
    baseline = logits[MODES[0]]
    comparisons, failures = {}, []

    def compare(actual, expected, label):
        actual, expected = actual.float(), expected.float()
        difference = (actual - expected).abs()
        rms = float(expected.square().mean().sqrt())
        stats = {"max_abs_error": float(difference.max()),
                 "reference_max_abs": float(expected.abs().max()),
                 "relative_rms_error": float(difference.square().mean().sqrt()) / max(rms, 1e-12),
                 "mismatched_elements": int((difference > atol + rtol * expected.abs()).sum()),
                 "elements": actual.numel()}
        try:
            torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        except AssertionError:
            failures.append(f"{label}: max_abs={stats['max_abs_error']:.6g}, "
                            f"mismatched={stats['mismatched_elements']}/{stats['elements']}")
        return stats

    for mode in MODES[1:]:
        logit_stats = compare(logits[mode], baseline, mode + " logits")
        kv_stats = {}
        if check_kv:
            for name, actual, expected in zip(("key", "value"), caches[mode], caches[MODES[0]]):
                kv_stats[name] = compare(actual, expected, mode + " " + name + " cache")
        comparisons[mode] = {
            "max_abs_logit_error": logit_stats["max_abs_error"], "logit_error": logit_stats,
            "greedy_tokens_match": bool(torch.equal(logits[mode].argmax(-1), baseline.argmax(-1))),
            "max_abs_kv_error": max((row["max_abs_error"] for row in kv_stats.values()), default=None),
            "kv_errors": kv_stats,
        }
    details = {"passed": not failures and all(row["greedy_tokens_match"] for row in comparisons.values()),
               "numerical_tolerance_passed": not failures, "atol": atol, "rtol": rtol,
               "kv_checked": check_kv, "comparisons": comparisons, "failures": failures,
               "last_token_argmax": {mode: value.argmax(-1).tolist() for mode, value in logits.items()}}
    if failures:
        raise PrefillValidationError(details)
    return details


def measure_forward(runner, case, mode):
    device = runner.cache.device
    cuda = device.type == "cuda"
    start = torch.cuda.Event(enable_timing=True) if cuda else None
    end = torch.cuda.Event(enable_timing=True) if cuda else None
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    synchronize(device)
    before_allocated = torch.cuda.memory_allocated(device) if cuda else None
    if cuda:
        start.record()
    began = time.perf_counter()
    execute_mode(runner, case, mode)
    if cuda:
        end.record()
    synchronize(device)
    wall_ms = (time.perf_counter() - began) * 1000.0
    peak_allocated = torch.cuda.max_memory_allocated(device) if cuda else None
    return {"wall_ms": wall_ms, "cuda_event_timeline_ms": start.elapsed_time(end) if cuda else None,
            "prefill_input_tokens": case["total_tokens"],
            "peak_allocated_bytes": peak_allocated,
            "additional_peak_allocated_bytes": peak_allocated - before_allocated if cuda else None,
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(device) if cuda else None}


def summarize_samples(samples):
    total_ms = sum(row["wall_ms"] for row in samples)
    return {
        "repetitions": len(samples),
        "wall_ms": {**percentiles(row["wall_ms"] for row in samples),
                    "mean": statistics.mean(row["wall_ms"] for row in samples)},
        "cuda_event_timeline_ms": {**percentiles(row["cuda_event_timeline_ms"] for row in samples),
                                   "mean": statistics.mean(row["cuda_event_timeline_ms"] for row in samples
                                                           if row["cuda_event_timeline_ms"] is not None)
                                   if any(row["cuda_event_timeline_ms"] is not None for row in samples) else None},
        "prefill_input_tokens_per_second": sum(row["prefill_input_tokens"] for row in samples) * 1000.0 / total_ms,
        "peak_allocated_bytes": max((row["peak_allocated_bytes"] for row in samples
                                     if row["peak_allocated_bytes"] is not None), default=None),
        "additional_peak_allocated_bytes": max((row["additional_peak_allocated_bytes"] for row in samples
                                                if row["additional_peak_allocated_bytes"] is not None), default=None),
        "peak_reserved_bytes": max((row["peak_reserved_bytes"] for row in samples
                                    if row["peak_reserved_bytes"] is not None), default=None),
    }


def run_case(runner, sequences, *, warmup=2, repetitions=5, atol=0.125, rtol=0.01, check_kv=True):
    if warmup < 0 or repetitions <= 0:
        raise ValueError("warmup must be nonnegative and repetitions must be positive")
    with prepared_case(runner, sequences) as case:
        correctness = correctness_checks(runner, case, atol=atol, rtol=rtol, check_kv=check_kv)
        for _ in range(warmup):
            for mode in MODES:
                execute_mode(runner, case, mode)
        synchronize(runner.cache.device)
        samples = {mode: [] for mode in MODES}
        order_log = []
        for repetition in range(repetitions):
            order = MODES[repetition % len(MODES):] + MODES[:repetition % len(MODES)]
            order_log.append(list(order))
            for mode in order:
                samples[mode].append(measure_forward(runner, case, mode))
        summaries = {mode: summarize_samples(rows) for mode, rows in samples.items()}
        times = {mode: sum(row["wall_ms"] for row in rows) for mode, rows in samples.items()}
        allowed = correctness["passed"] and correctness["kv_checked"]
        return {
            "batch_size": len(sequences), "prompt_lengths": case["lengths"],
            "total_prompt_tokens": case["total_tokens"], "prompt_token_sha256": sha256_json(sequences),
            "query_starts": [0] * len(sequences), "prefix_cache": "disabled",
            "allocated_request_pages": sum(map(len, case["pages"])),
            "kv_pool_bytes": runner.cache.memory_bytes,
            "correctness": correctness, "comparison_allowed": allowed,
            "comparison_blockers": (["kv_not_checked"] if not correctness["kv_checked"] else [])
                                   + (["greedy_tokens_differ"] if not correctness["passed"] else []),
            "samples": samples, "summary": summaries, "measurement_order": order_log,
            "sequential_full_over_selected_wall_ratio": times[MODES[0]] / times[MODES[1]] if allowed else None,
            "sequential_selected_over_ragged_wall_ratio": times[MODES[1]] / times[MODES[2]] if allowed else None,
        }


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", default=str(ROOT / "models" / "Qwen3-0.6B"))
    result.add_argument("--device", default="cuda")
    result.add_argument("--dtype", choices=("fp16", "bf16", "fp32"), default="fp16")
    result.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4])
    result.add_argument("--prompt-lengths", type=int, nargs="+", default=[32, 64], help="maximum length in each ragged batch")
    result.add_argument("--prompt", default="Explain why an inference engine uses a paged KV cache. ")
    result.add_argument("--warmup", type=int, default=2)
    result.add_argument("--repetitions", type=int, default=5)
    result.add_argument("--block-size", type=int, default=16)
    result.add_argument("--num-blocks", type=int, default=128)
    result.add_argument("--prefill-backend", choices=("auto", "math", "cuda", "dandan"), default="cuda")
    result.add_argument("--elementwise-backend", choices=("torch", "cuda"), default="cuda")
    result.add_argument("--data-backend", choices=("torch", "cuda"), default="cuda")
    result.add_argument("--linear-backend", choices=("torch", "cuda", "dandan", "dandan_cuda"), default="cuda")
    result.add_argument("--fuse-projections", action=argparse.BooleanOptionalAction, default=True)
    result.add_argument("--atol", type=float)
    result.add_argument("--rtol", type=float, default=0.01)
    result.add_argument("--skip-kv-check", action="store_true")
    result.add_argument("--output", type=Path, default=ROOT / "benchmarks" / "results" / "ragged_prefill.json")
    return result


def load_runner(args):
    return NativeQwen3Runner.from_pretrained(args.model, device=args.device,
        dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype],
        prefill_backend=args.prefill_backend, elementwise_backend=args.elementwise_backend,
        data_backend=args.data_backend, linear_backend=args.linear_backend,
        fuse_projections=args.fuse_projections, num_blocks=args.num_blocks, block_size=args.block_size,
        local_files_only=True)


def main(argv=None):
    arg_parser = parser()
    args = arg_parser.parse_args(argv)
    if (min(args.batch_sizes + args.prompt_lengths + [args.block_size, args.num_blocks, args.repetitions]) <= 0
            or args.warmup < 0 or not math.isfinite(args.rtol) or args.rtol < 0
            or (args.atol is not None and (not math.isfinite(args.atol) or args.atol < 0))):
        arg_parser.error("lengths, capacities and repetitions must be positive; warmup and tolerances nonnegative")
    atol = args.atol if args.atol is not None else (0.25 if args.dtype == "bf16" else 0.125 if args.dtype == "fp16" else 1e-4)
    payload = {"kind": "ragged_cold_prefill_forward", "status": "failed", "cases": [],
               "configuration": {**vars(args), "output": str(args.output)},
               "torch": torch.__version__, "torch_cuda": torch.version.cuda,
               "timing_scope": "Prebuilt PrefillBatch forwards only: B1/all query logits, B1/last query logits, or one packed batch/last rows. Includes forward validation, selected-row index validation/device index-tensor creation, output copies and completion synchronization. Excludes model load, tokenization, batch metadata construction, host selection planning, correctness checks, warmup, fingerprint/GPU snapshots and cleanup. CUDA Events measure the dispatch timeline including launch gaps, not pure kernel instructions. No decode, scheduler, sampling, generation TPS or HTTP latency is measured.",
               "memory_scope": "Allocated peaks include the loaded model, preallocated KV pool and output buffers. Additional allocated peak is relative to pre-forward allocation. Reserved peaks include allocator caching from setup, correctness and previous modes; they are not isolated process-memory footprints.",
               "limitations": ["Cold prefill only; every query starts at position zero.",
                               "All modes use the same new prefill attention path; sequential_full is a prebuilt B1 baseline, not an exact legacy implementation.",
                               "GPU snapshots cannot establish exclusive hardware use, especially under Windows WDDM."]}
    if args.skip_kv_check:
        payload["limitations"].append("KV validation was explicitly skipped: raw timings are exploratory, comparison_allowed is false, and no accepted comparison ratio is calculated.")
    runner, failure = None, None
    try:
        payload["source_start"], payload["gpu_start"] = source_fingerprint(), gpu_snapshot()
        runner = load_runner(args)
        payload["backend_start"] = runner.model.backend_report()
        for maximum in args.prompt_lengths:
            for batch in args.batch_sizes:
                sequences = prompt_tokens(runner, ragged_lengths(batch, maximum), args.prompt)
                payload["cases"].append(run_case(runner, sequences, warmup=args.warmup, repetitions=args.repetitions,
                                                 atol=atol, rtol=args.rtol, check_kv=not args.skip_kv_check))
        payload["backend_end"] = runner.model.backend_report()
        payload["status"] = ("output_mismatch" if not all(case["correctness"]["passed"] for case in payload["cases"])
                             else "validation_incomplete" if args.skip_kv_check else "passed")
    except BaseException as error:
        failure = error
        payload["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        if isinstance(error, PrefillValidationError):
            payload["error"]["validation"] = error.details
    finally:
        if runner is not None:
            try:
                runner.close()
            except Exception as error:
                payload["close_error"] = {"type": type(error).__name__, "message": str(error)}
                payload["status"] = "failed"
                failure = failure or error
        try:
            payload["source_end"], payload["gpu_end"] = source_fingerprint(), gpu_snapshot()
        except Exception as error:
            payload["metadata_end_error"] = {"type": type(error).__name__, "message": str(error)}
            payload["status"] = "failed"
            failure = failure or error
        start_hash = payload.get("source_start", {}).get("source_tree_sha256")
        end_hash = payload.get("source_end", {}).get("source_tree_sha256")
        payload["source_unchanged"] = start_hash == end_hash if start_hash is not None and end_hash is not None else None
        payload["backend_unchanged"] = (payload["backend_start"] == payload["backend_end"]
                                        if "backend_start" in payload and "backend_end" in payload else None)
        payload["source_and_backend_unchanged"] = (
            payload["source_unchanged"] is True and payload["backend_unchanged"] is True)
        if not payload["source_and_backend_unchanged"] or payload["status"] not in {"passed", "validation_incomplete"}:
            if payload["status"] == "passed":
                payload["status"] = "source_or_backend_changed"
            for case in payload["cases"]:
                case["comparison_allowed"] = False
                blocker = "source_or_backend_unverified" if not payload["source_and_backend_unchanged"] else "run_failed"
                case["comparison_blockers"].append(blocker)
                case["sequential_full_over_selected_wall_ratio"] = None
                case["sequential_selected_over_ragged_wall_ratio"] = None
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": payload["status"], "cases": len(payload["cases"]), "output": str(args.output)}))
    if failure is not None:
        raise failure
    return payload


if __name__ == "__main__":
    main()
