"""Profile a warmed native runtime; annotations are confined to this script.

Use --tool torch for operator tables/Chrome traces, or --tool nvtx under nsys.
Profiling overhead means the collected wall time is not a throughput benchmark.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import contextmanager, ExitStack, nullcontext
import functools
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from mini_engine.engine.request import GenerationRequest
from mini_engine.model import NativeQwen3Runner
from mini_engine.runtime import configure_local_cache


@contextmanager
def marked_range(name, *, nvtx=False, record=True):
    with torch.profiler.record_function(name) if record else nullcontext():
        if nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if nvtx:
                torch.cuda.nvtx.range_pop()


@contextmanager
def instrument_runtime(runtime, *, nvtx=False, record=True, layers=False):
    """Restore every patched method/function, including on execution errors."""
    import mini_engine.model.native_qwen3 as model_ops

    with ExitStack() as stack:
        def patch(obj, attribute, label):
            original = getattr(obj, attribute)
            had_attribute = attribute in vars(obj)
            def restore():
                if had_attribute:
                    setattr(obj, attribute, original)
                else:
                    delattr(obj, attribute)
            stack.callback(restore)
            @functools.wraps(original)
            def wrapped(*args, **kwargs):
                with marked_range(label, nvtx=nvtx, record=record):
                    return original(*args, **kwargs)
            setattr(obj, attribute, wrapped)

        for obj, method, label in (
            (runtime, "prepare_waiting", "engine/prefix_lookup"),
            (runtime.scheduler, "schedule", "engine/schedule"),
            (runtime, "prefill_step", "engine/prefill"),
            (runtime, "decode_step", "engine/decode"),
            (runtime, "_sample", "engine/sampling"),
        ):
            patch(obj, method, label)
        if runtime.graph_manager is not None:
            for bucket in runtime.graph_manager.buckets.values():
                patch(bucket, "replay", "engine/graph_buffers_and_replay")
                patch(bucket, "replay_metadata", "engine/graph_buffers_and_replay")
        for function in ("embedding_lookup", "rms_norm", "residual_rms_norm", "residual_add",
                         "silu_and_mul", "apply_rope_backend", "write_kv_cache_batch",
                         "write_kv_cache_slots", "paged_prefill_attention",
                         "prefill_attention", "paged_attention"):
            patch(model_ops, function, "op/" + function)
        selected = []
        for name, module in runtime.model.named_modules():
            if (hasattr(module, "in_features") and hasattr(module, "out_features")
                    and not any(name.startswith(parent + ".") for parent in selected)):
                selected.append(name)
                patch(module, "forward", "linear/" + name if layers else "op/linear")
        yield


def generate(runtime, prompts, output_tokens, *, markers=False, nvtx=False, record=True):
    requests = [GenerationRequest(f"profile-{index}", list(ids), max_new_tokens=output_tokens)
                for index, ids in enumerate(prompts)]
    for request in requests:
        runtime.scheduler.add_request(request)
    limit = sum(len(ids) + output_tokens + 1 for ids in prompts) * 4
    steps, started = 0, time.perf_counter()
    try:
        while runtime.scheduler.waiting or runtime.scheduler.running:
            if steps >= limit:
                raise RuntimeError("runtime did not finish within the profiling step limit")
            with marked_range("engine/step", nvtx=nvtx, record=record) if markers else nullcontext():
                runtime.step()
            steps += 1
        if runtime.cache.device.type == "cuda":
            torch.cuda.synchronize(runtime.cache.device)
        return {"steps": steps, "wall_ms": (time.perf_counter() - started) * 1000,
                "output_tokens": sum(len(r.output_token_ids) for r in requests),
                "output_sha256": hashlib.sha256(json.dumps([r.output_token_ids for r in requests]).encode()).hexdigest()}
    finally:
        for request in requests:
            if not request.is_finished:
                runtime.scheduler.abort(request.request_id)
        runtime.prefix_cache.clear()


def event_row(event):
    return {"name": event.key, "calls": event.count,
            "self_cpu_us": event.self_cpu_time_total,
            "cpu_total_us": event.cpu_time_total,
            "self_device_us": getattr(event, "self_device_time_total", 0),
            "device_total_us": getattr(event, "device_time_total", 0),
            "device_type": str(getattr(event, "device_type", "unknown"))}


def main():
    configure_local_cache()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(ROOT / "models" / "Qwen3-0.6B"))
    parser.add_argument("--tool", choices=["torch", "nvtx"], default="torch")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--cuda-graph", action="store_true")
    parser.add_argument("--cuda-profiler-api", action="store_true", help="Start/stop nsys collection after warmup; use with --tool nvtx")
    parser.add_argument("--prefill-backend", choices=["auto", "math", "cuda", "dandan"], default="cuda")
    parser.add_argument("--linear-backend", choices=["torch", "cuda", "dandan", "dandan_cuda"], default="cuda")
    parser.add_argument("--quantization", choices=["none", "int4"], default="none")
    parser.add_argument("--prefill-chunk-size", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--shapes", action="store_true")
    parser.add_argument("--memory", action="store_true")
    parser.add_argument("--stacks", action="store_true")
    parser.add_argument("--layers", action="store_true", help="Separate projection scopes by layer; eager path only")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "build" / "profiles" / "torch_eager")
    args = parser.parse_args()
    if min(args.batch_size, args.prompt_tokens, args.max_new_tokens, args.warmup,
           args.prefill_chunk_size, args.num_blocks, args.block_size) <= 0:
        parser.error("batch, prompt, output, warmup, chunk and page limits must be positive")
    if args.cuda_profiler_api and (args.tool != "nvtx" or not args.device.startswith("cuda")):
        parser.error("--cuda-profiler-api requires --tool nvtx and CUDA")
    if args.quantization == "int4" and args.dtype != "fp16":
        parser.error("W4A16 requires --dtype fp16")
    if args.cuda_graph and (not args.device.startswith("cuda") or args.dtype == "fp32"):
        parser.error("CUDA Graph requires CUDA FP16/BF16")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    from mini_engine.config import QuantConfig
    is_cuda = args.device.startswith("cuda")
    runner = NativeQwen3Runner.from_pretrained(
        args.model, device=args.device,
        dtype={"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype],
        prefill_backend=args.prefill_backend, decode_backend="cuda" if is_cuda else "torch",
        elementwise_backend="cuda" if is_cuda else "torch", data_backend="cuda" if is_cuda else "torch",
        linear_backend=args.linear_backend, fuse_projections=True, local_files_only=True,
        num_blocks=args.num_blocks, block_size=args.block_size,
        quant_backend="dandan" if args.quantization == "int4" else "reference",
        quantization=QuantConfig("int4", "fp16") if args.quantization == "int4" else None,
    )
    runtime = None
    try:
        required_pages = (args.prompt_tokens + args.max_new_tokens - 2) // args.block_size + 1
        block_bucket = max(8, 1 << (required_pages - 1).bit_length())
        batch_bucket = 1 << (args.batch_size - 1).bit_length()
        runtime = runner.create_batch_runtime(
            use_cuda_graph=args.cuda_graph, enable_prefix_cache=False,
            prefill_chunk_size=args.prefill_chunk_size,
            graph_batch_sizes=(batch_bucket,), graph_block_buckets=(block_bucket,),
        )
        runtime.eos_token_ids.clear()
        source = runner.tokenizer.apply_chat_template([{"role": "user", "content": "Explain KV Cache in one sentence."}],
            tokenize=True, add_generation_prompt=True, enable_thinking=False)
        prompt = (source * ((args.prompt_tokens + len(source) - 1) // len(source)))[:args.prompt_tokens]
        prompts = [prompt for _ in range(args.batch_size)]
        backend_start = runner.model.backend_report()
        warmup = [generate(runtime, prompts, args.max_new_tokens) for _ in range(args.warmup)]
        activities = [torch.profiler.ProfilerActivity.CPU]
        supported = torch.profiler.supported_activities()
        if is_cuda and torch.profiler.ProfilerActivity.CUDA in supported:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        profiler = torch.profiler.profile(activities=activities, record_shapes=args.shapes,
            profile_memory=args.memory, with_stack=args.stacks) if args.tool == "torch" else nullcontext()
        api = torch.cuda.cudart() if args.cuda_profiler_api else None
        try:
            if api is not None:
                torch.cuda.check_error(api.cudaProfilerStart())
            with profiler as prof, instrument_runtime(runtime, nvtx=is_cuda, record=args.tool == "torch", layers=args.layers):
                with marked_range("engine/generation", nvtx=is_cuda, record=args.tool == "torch"):
                    result = generate(runtime, prompts, args.max_new_tokens, markers=True,
                                      nvtx=is_cuda, record=args.tool == "torch")
        finally:
            if api is not None:
                torch.cuda.check_error(api.cudaProfilerStop())
        metadata = {"scope": "Instrumented warmed runtime, not a latency benchmark; excludes load/tokenization/capture.",
            "configuration": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "os": platform.platform(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name() if is_cuda else None,
            "supported_activities": [str(activity) for activity in supported],
            "captured_graph_buckets": [vars(shape) for shape, bucket in runtime.graph_manager.buckets.items() if bucket.captured] if runtime.graph_manager else [],
            "warmup_median_ms_not_a_benchmark": statistics.median(row["wall_ms"] for row in warmup),
            "profiled_generation": result, "output_matches_warmup": all(row["output_sha256"] == result["output_sha256"] for row in warmup),
            "backend_report_start": backend_start, "backend_report_end": runner.model.backend_report(),
            "notes": ["CUDA Graph replay has no Python per-layer execution. Use nsys node tracing to inspect captured kernels.",
                      "CPU and device durations overlap; inclusive parent ranges double count children.",
                      "Synthetic repeated tokenized prompt, fixed output count, prefix reuse disabled."]}
        if args.tool == "torch":
            prof.export_chrome_trace(str(args.output_dir / "trace.json"))
            averages = prof.key_averages(group_by_input_shape=args.shapes)
            table = averages.table(sort_by="self_cpu_time_total", row_limit=30) + "\n" + averages.table(sort_by="self_device_time_total", row_limit=30)
            (args.output_dir / "operators.txt").write_text(table, encoding="utf-8")
            rows = [event_row(event) for event in averages]
            metadata["top_cpu"] = sorted(rows, key=lambda row: row["self_cpu_us"], reverse=True)[:30]
            metadata["top_device"] = sorted(rows, key=lambda row: row["self_device_us"], reverse=True)[:30]
            trace = json.loads((args.output_dir / "trace.json").read_text(encoding="utf-8"))
            kernels = [event for event in trace.get("traceEvents", []) if event.get("cat", "").lower() == "kernel" and "dur" in event]
            metadata["cuda_kernel_events_in_trace"] = len(kernels)
            metadata["device_timeline_available"] = metadata["cuda_kernel_events_in_trace"] > 0
            # CUDA user annotations also appear as device events. They enclose
            # physical kernels, so summing all key_averages device times would
            # double count. Keep a separate table of physical kernel durations.
            totals = defaultdict(lambda: {"calls": 0, "total_us": 0.0})
            for kernel in kernels:
                totals[kernel["name"]]["calls"] += 1
                totals[kernel["name"]]["total_us"] += kernel["dur"]
            metadata["physical_kernel_total_us"] = sum(value["total_us"] for value in totals.values())
            metadata["top_physical_kernels"] = sorted(
                [{"name": name, **value} for name, value in totals.items()],
                key=lambda value: value["total_us"], reverse=True)[:30]
            print(table)
        (args.output_dir / "summary.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote profiling artifacts to {args.output_dir}")
    finally:
        # instrument_runtime/profiler contexts unwind before resource cleanup.
        try:
            if runtime is not None:
                runtime.close()
        finally:
            runner.close()



if __name__ == "__main__":
    main()
