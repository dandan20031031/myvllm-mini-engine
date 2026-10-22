"""Benchmark native continuous batching with eager or CUDA Graph decode.

The benchmark intentionally times the scheduler-driven runtime rather than the
single-request runner. Warmup exercises graph capture (when enabled), and each
measured repetition starts with fresh requests. Prefix entries are cleared
between repetitions so they do not silently change the workload.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import statistics
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_engine.engine.request import GenerationRequest
from mini_engine.model import NativeQwen3Runner
from mini_engine.runtime import configure_local_cache


def synchronize(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def encode_prompts(runner, prompts: list[str]) -> list[list[int]]:
    encoded = []
    for prompt in prompts:
        ids = runner.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_tensors="pt",
        )[0]
        encoded.append([int(token) for token in ids.tolist()])
    return encoded


def run_once(runtime, prompt_ids: list[list[int]], max_new_tokens: int, device: str):
    requests = [
        GenerationRequest(
            request_id=f"bench-{index}",
            prompt_token_ids=ids,
            max_new_tokens=max_new_tokens,
        )
        for index, ids in enumerate(prompt_ids)
    ]
    for request in requests:
        runtime.scheduler.add_request(request)
    synchronize(device)
    start = time.perf_counter()
    steps = 0
    while runtime.scheduler.waiting or runtime.scheduler.running:
        runtime.step()
        steps += 1
    synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    output_tokens = sum(len(request.output_token_ids) for request in requests)
    runtime.prefix_cache.clear()
    return {
        "total_ms": elapsed_ms,
        "output_tokens": output_tokens,
        "steps": steps,
        "tokens_per_second": output_tokens / (elapsed_ms / 1000.0),
    }


def run_mode(args, use_cuda_graph: bool):
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    runner = NativeQwen3Runner.from_pretrained(
        args.model,
        device=args.device,
        dtype=dtype,
        prefill_backend=args.prefill_backend,
        decode_backend=args.decode_backend,
        elementwise_backend=args.elementwise_backend,
        data_backend=args.data_backend,
        linear_backend=args.linear_backend,
        num_blocks=args.num_blocks,
        block_size=args.block_size,
        local_files_only=args.local_files_only,
        fuse_projections=args.fuse_projections,
    )
    runtime = runner.create_batch_runtime(
        use_cuda_graph=use_cuda_graph,
        graph_batch_sizes=(1, 2, 4, 8, 16),
        graph_block_buckets=(8, 16, 32, 64),
    )
    prompt_ids = encode_prompts(runner, args.prompts)
    for _ in range(args.warmup):
        run_once(runtime, prompt_ids, args.max_new_tokens, args.device)
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    samples = [run_once(runtime, prompt_ids, args.max_new_tokens, args.device) for _ in range(args.repetitions)]
    payload = {
        "mode": "cuda_graph" if use_cuda_graph else "eager",
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "dtype": args.dtype,
        "batch_size": len(args.prompts),
        "max_new_tokens": args.max_new_tokens,
        "block_size": args.block_size,
        "backends": {
            "prefill": args.prefill_backend,
            "decode": args.decode_backend,
            "elementwise": args.elementwise_backend,
            "data": args.data_backend,
            "linear": args.linear_backend,
        },
        "projection_fusion": args.fuse_projections,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "samples": samples,
        "median_total_ms": statistics.median(row["total_ms"] for row in samples),
        "median_tokens_per_second": statistics.median(row["tokens_per_second"] for row in samples),
        "median_output_tokens": statistics.median(row["output_tokens"] for row in samples),
        "peak_memory_mb": (
            torch.cuda.max_memory_allocated(torch.device(args.device)) / (1024 * 1024)
            if torch.cuda.is_available() and str(args.device).startswith("cuda") else None
        ),
    }
    del runtime, runner
    gc.collect()
    if torch.cuda.is_available() and str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--decode-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--elementwise-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--data-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--linear-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--prefill-backend", choices=["auto", "math", "cuda"], default="cuda")
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "results" / "native_batch.json")
    args = parser.parse_args()
    configure_local_cache()
    if not args.prompts:
        args.prompts = [
            "Explain KV Cache in one sentence.",
            "Explain continuous batching in one sentence.",
            "Explain CUDA Graph in one sentence.",
        ]
    if args.max_new_tokens <= 0 or args.warmup < 0 or args.repetitions <= 0:
        raise SystemExit("max-new-tokens > 0, warmup >= 0 and repetitions > 0 are required")
    if args.local_files_only and not Path(args.model).is_dir():
        raise SystemExit("local model directory is missing; benchmark not run")
    results = {
        "eager": run_mode(args, use_cuda_graph=False),
        "cuda_graph": run_mode(args, use_cuda_graph=True),
    }
    eager_tps = results["eager"]["median_tokens_per_second"]
    graph_tps = results["cuda_graph"]["median_tokens_per_second"]
    payload = {
        "results": results,
        "graph_speedup": graph_tps / eager_tps if eager_tps else None,
        "note": "Prefill remains sequential; Graph mode captures the batched one-token decode path. Results are not vLLM comparisons.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
