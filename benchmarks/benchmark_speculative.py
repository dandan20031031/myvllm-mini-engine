"""Pair eager greedy with prompt-lookup speculation on identical fresh requests.

Use --toy for a deterministic CPU acceptance/rollback demonstration. Omit --toy
and supply a local checkpoint to measure a real Qwen3 model. Toy latency never
represents model or service latency. Both real paths are single-request eager.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
from types import SimpleNamespace

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_engine.config import QuantConfig
from mini_engine.model import NativeQwen3Runner
from mini_engine.runtime import configure_local_cache
from mini_engine.speculative import generate_speculative


class ToyTokenizer:
    eos_token_id = None

    def apply_chat_template(self, *args, **kwargs):
        return torch.tensor([list(range(32)) + [0]])

    def decode(self, token_ids, **kwargs):
        return " ".join(map(str, token_ids))


class ToyTarget(torch.nn.Module):
    """A cyclic next-token rule, with actual writes to physical KV slots."""

    decode_backend = "torch"
    config = SimpleNamespace(max_position_embeddings=4096)

    def create_cache(self, num_blocks, block_size):
        return SimpleNamespace(device=torch.device("cpu"), block_size=block_size, tokens=torch.full((num_blocks * block_size,), -1))

    def forward(self, ids, cache, block_ids, position=0):
        for offset, token in enumerate(ids):
            absolute = position + offset
            cache.tokens[block_ids[absolute // cache.block_size] * cache.block_size + absolute % cache.block_size] = token
        return torch.nn.functional.one_hot((ids + 1) % 32, num_classes=32).float() * 100.0


def result_payload(result):
    return {
        "token_ids": result.token_ids, "finish_reason": result.finish_reason,
        "output_tokens": len(result.token_ids),
        "ttft_ms": result.ttft_ms, "tpot_ms": result.tpot_ms,
        "total_latency_ms": result.total_latency_ms,
        "tokens_per_second": len(result.token_ids) * 1000.0 / result.total_latency_ms,
        "stats": getattr(result, "stats", {"target_forward_calls": len(result.token_ids)}),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--toy", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-draft-tokens", type=int, default=4)
    parser.add_argument("--min-ngram-size", type=int, default=2)
    parser.add_argument("--max-ngram-size", type=int, default=4)
    parser.add_argument("--adaptive-draft", action="store_true")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--prefill-backend", choices=["auto", "math", "cuda", "dandan"], default="cuda")
    parser.add_argument("--decode-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--elementwise-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--data-backend", choices=["torch", "cuda"], default="cuda")
    parser.add_argument("--linear-backend", choices=["torch", "cuda", "dandan", "dandan_cuda"], default="cuda")
    parser.add_argument("--quantization", choices=["none", "int4"], default="none")
    parser.add_argument("--quant-backend", choices=["reference", "dandan"], default="reference")
    parser.add_argument("--quant-group-size", type=int, default=128)
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "results" / "speculative.json")
    args = parser.parse_args()
    if args.max_new_tokens <= 0 or args.warmup < 0 or args.repetitions <= 0:
        parser.error("max-new-tokens > 0, warmup >= 0 and repetitions > 0 are required")
    if args.max_draft_tokens < 0 or args.min_ngram_size <= 0 or args.max_ngram_size < args.min_ngram_size:
        parser.error("max-draft-tokens >= 0 and 0 < min-ngram-size <= max-ngram-size are required")
    if not args.toy and (not args.model or not Path(args.model).is_dir()):
        parser.error("provide --toy or a complete local --model directory")
    configure_local_cache()
    torch.manual_seed(41)
    if args.toy:
        runner = NativeQwen3Runner(ToyTarget(), ToyTokenizer(), num_blocks=512, block_size=16)
    else:
        runner = NativeQwen3Runner.from_pretrained(
            args.model, device=args.device,
            dtype={"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype],
            prefill_backend=args.prefill_backend, decode_backend=args.decode_backend,
            elementwise_backend=args.elementwise_backend, data_backend=args.data_backend,
            linear_backend=args.linear_backend, local_files_only=True,
            fuse_projections=args.fuse_projections,
            quant_backend=args.quant_backend,
            quantization=QuantConfig("int4", args.dtype, args.quant_group_size) if args.quantization == "int4" else None,
        )
    dependency_start = runner.model.backend_report() if hasattr(runner.model, "backend_report") else None
    prompts = args.prompts or (["cyclic CPU target; the fixed 33-token history repeats"] if args.toy else [
        "Repeat this sentence exactly three times: The cat sat on the mat.",
        "用一句话解释 KV Cache。",
    ])
    spec_options = dict(
        max_draft_tokens=args.max_draft_tokens, min_ngram_size=args.min_ngram_size,
        max_ngram_size=args.max_ngram_size, adaptive_draft=args.adaptive_draft,
    )
    previous_eos = runner.eos_token_ids
    samples = []
    try:
        if args.ignore_eos:
            runner.eos_token_ids = []
        for prompt in prompts:
            for _ in range(args.warmup):
                runner.generate_incremental(prompt, args.max_new_tokens)
                generate_speculative(runner, prompt, args.max_new_tokens, **spec_options)
            for repetition in range(args.repetitions):
                calls = {
                    "eager": lambda: runner.generate_incremental(prompt, args.max_new_tokens),
                    "speculative": lambda: generate_speculative(runner, prompt, args.max_new_tokens, **spec_options),
                }
                order = ("eager", "speculative") if repetition % 2 == 0 else ("speculative", "eager")
                results = {mode: result_payload(calls[mode]()) for mode in order}
                samples.append({
                    "prompt": prompt, "repetition": repetition,
                    "execution_order": list(order),
                    "greedy_output_match": results["eager"]["token_ids"] == results["speculative"]["token_ids"],
                    **results,
                })
    finally:
        runner.eos_token_ids = previous_eos
    dependency_end = runner.model.backend_report() if hasattr(runner.model, "backend_report") else None
    same_output = all(sample["greedy_output_match"] for sample in samples)
    summary = []
    for prompt in prompts:
        group = [row for row in samples if row["prompt"] == prompt]
        original_ms = statistics.median(row["eager"]["total_latency_ms"] for row in group)
        spec_ms = statistics.median(row["speculative"]["total_latency_ms"] for row in group)
        proposed = sum(row["speculative"]["stats"]["proposed_tokens"] for row in group)
        accepted = sum(row["speculative"]["stats"]["accepted_tokens"] for row in group)
        summary.append({
            "prompt": prompt, "median_eager_ms": original_ms, "median_speculative_ms": spec_ms,
            "observed_latency_ratio": original_ms / spec_ms if same_output else None,
            "acceptance_rate": accepted / proposed if proposed else 0.0,
            "median_proposal_hit_rate": statistics.median(row["speculative"]["stats"]["proposal_hit_rate"] for row in group),
            "median_eager_target_calls": statistics.median(row["eager"]["stats"]["target_forward_calls"] for row in group),
            "median_speculative_target_calls": statistics.median(row["speculative"]["stats"]["target_forward_calls"] for row in group),
        })
    payload = {
        "status": "passed" if same_output else "greedy_output_mismatch",
        "model": "deterministic_cpu_toy" if args.toy else Path(args.model).name,
        "toy": args.toy, "device": "cpu" if args.toy else args.device,
        "dtype": "synthetic_fp32_logits" if args.toy else args.dtype,
        "pytorch": str(torch.__version__), "torch_cuda": torch.version.cuda,
        "max_new_tokens": args.max_new_tokens, "ignore_eos": args.ignore_eos,
        "warmup": args.warmup, "repetitions": args.repetitions,
        "speculative_options": spec_options,
        "backend_report_start": dependency_start, "backend_report_end": dependency_end,
        "library_files_unchanged_during_run": dependency_start == dependency_end,
        "summary": summary, "samples": samples,
        "timing_scope": "Per-request wall time from before prefill until the final output bundle is ready; includes draft lookup, target execution and acceptance/rollback logic. Excludes weight loading, tokenization and text decoding. TPOT is an average over emitted output tokens, not a streaming delivery interval for bundled speculative tokens.",
        "limitations": [
            "Greedy prompt lookup only; no trained draft model, EAGLE, MTP, or random-sampling correction is integrated.",
            "Both paths are single-request eager; this is not a decode CUDA Graph, continuous-batching or HTTP service comparison.",
            "Toy acceptance and latency do not predict real-model speedup. Real gains depend on prompt repetition and target suffix cost.",
            "Unchanged dependency files do not prove an uncontended GPU. Near-tied logits can produce suffix-versus-token argmax differences.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if same_output else 1)


if __name__ == "__main__":
    main()
