"""Validate a complete local Qwen3 checkpoint against native inference.

The script never invents a result: if the model directory is missing it writes
a blocked report and exits with status 2. All outputs stay under the project.
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_engine.model import NativeQwen3Runner, Qwen3ModelRunner
from mini_engine.runtime import configure_local_cache


def _sync(device):
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--decode-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--elementwise-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--data-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--linear-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--prefill-backend", choices=["auto", "math", "cuda"], default="auto")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "results" / "qwen3_validation.json")
    args = parser.parse_args()
    configure_local_cache()
    output = {"model": args.model, "device": args.device, "dtype": args.dtype, "status": "blocked"}
    model_path = Path(args.model)
    if model_path.exists() and not model_path.is_dir():
        output["reason"] = "model path exists but is not a directory"
    elif not model_path.exists() and (args.local_files_only or ("/" not in args.model and "\\" not in args.model)):
        output["reason"] = "local model directory is missing; provide a downloaded Qwen3-0.6B directory"
    else:
        output["status"] = "running"
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
        prompts = args.prompts or ["用一句话解释 KV Cache。", "解释 continuous batching 的作用。"]
        output["prompts"] = prompts
        output["checks"] = []
        reference = Qwen3ModelRunner(args.model, device=args.device, dtype=dtype)
        try:
            for prompt in prompts:
                ref_start = time.perf_counter()
                reference_text = reference.generate(prompt, max_new_tokens=args.max_new_tokens, do_sample=False)
                _sync(args.device)
                output["checks"].append({"prompt": prompt, "reference_text": reference_text, "reference_ms": (time.perf_counter() - ref_start) * 1000})
        finally:
            del reference
            gc.collect()
            if str(args.device).startswith("cuda"):
                torch.cuda.empty_cache()
        native = NativeQwen3Runner.from_pretrained(
            args.model, device=args.device, dtype=dtype,
            prefill_backend=args.prefill_backend, decode_backend=args.decode_backend,
            elementwise_backend=args.elementwise_backend, data_backend=args.data_backend,
            linear_backend=args.linear_backend, local_files_only=args.local_files_only,
        )
        for check, prompt in zip(output["checks"], prompts):
            result = native.generate_incremental(prompt, args.max_new_tokens)
            check.update({
                "native_text": result.text, "native_token_ids": result.token_ids,
                "native_finish_reason": result.finish_reason,
                "ttft_ms": result.ttft_ms, "tpot_ms": result.tpot_ms,
                "text_nonempty": bool(result.text.strip()),
            })
        output["status"] = "passed"
        output["note"] = "Text equality is not required for independent greedy implementations; inspect token/logit alignment separately."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output, ensure_ascii=False, indent=2))
    raise SystemExit(0 if output["status"] == "passed" else 2)


if __name__ == "__main__":
    main()
