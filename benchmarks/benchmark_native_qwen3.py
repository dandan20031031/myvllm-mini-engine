"""End-to-end native Qwen3 benchmark with explicit backend metadata."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_engine.model import NativeQwen3Runner
from mini_engine.runtime import configure_local_cache


def sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


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
    parser.add_argument("--prefill-backend", choices=["auto", "math", "cuda"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "results" / "native_qwen3.json")
    args = parser.parse_args()
    configure_local_cache()
    model_path = Path(args.model)
    if args.local_files_only and not model_path.is_dir():
        raise SystemExit("local model directory is missing; benchmark not run")
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    prompts = args.prompts or ["用一句话解释 KV Cache。", "解释 continuous batching。"]
    runner = NativeQwen3Runner.from_pretrained(
        args.model, device=args.device, dtype=dtype,
        prefill_backend=args.prefill_backend, decode_backend=args.decode_backend,
        elementwise_backend=args.elementwise_backend, data_backend=args.data_backend,
        linear_backend=args.linear_backend, local_files_only=args.local_files_only,
    )
    for prompt in prompts:
        for _ in range(args.warmup):
            runner.generate_incremental(prompt, args.max_new_tokens)
    rows = []
    for prompt in prompts:
        samples = []
        for _ in range(args.repetitions):
            sync(args.device)
            start = time.perf_counter()
            result = runner.generate_incremental(prompt, args.max_new_tokens)
            sync(args.device)
            total_ms = (time.perf_counter() - start) * 1000
            samples.append({"total_ms": total_ms, "ttft_ms": result.ttft_ms, "tpot_ms": result.tpot_ms, "output_tokens": len(result.token_ids), "finish_reason": result.finish_reason})
        rows.append({"prompt": prompt, "samples": samples, "median_total_ms": statistics.median(item["total_ms"] for item in samples), "median_ttft_ms": statistics.median(item["ttft_ms"] for item in samples), "median_tpot_ms": statistics.median(item["tpot_ms"] for item in samples), "median_output_tokens": statistics.median(item["output_tokens"] for item in samples)})
    payload = {"model": args.model, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda, "dtype": args.dtype, "backends": {"prefill": args.prefill_backend, "decode": args.decode_backend, "elementwise": args.elementwise_backend, "data": args.data_backend, "linear": args.linear_backend}, "warmup": args.warmup, "repetitions": args.repetitions, "rows": rows, "note": "Requires complete local weights; results are native single-request measurements, not vLLM comparisons."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
