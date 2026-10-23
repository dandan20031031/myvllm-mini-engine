"""Validate dandan FP16 and W4A16 on a complete local Qwen3 checkpoint.

Compare FP16 projections against the engine's cuBLAS path. For W4A16, compare
against the *same packed weights* dequantized and rounded to FP16, with FP32
matmul accumulation. Separately report error introduced by quantization. These
are short numerical checks, not a model-quality evaluation or a performance test.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path
import sys

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mini_engine.config import QuantConfig
from mini_engine.model import NativeQwen3Runner
from mini_engine.runtime import configure_local_cache


class DequantizedFP16Oracle(nn.Module):
    """Match Tensor Core's FP16 operands without retaining packed weights."""

    def __init__(self, quantized):
        super().__init__()
        self.register_buffer("weight", quantized.dequantize().half())
        self.register_buffer("bias", None if quantized.bias is None else quantized.bias.detach().clone())

    def forward(self, x):
        # Quantization rounds the dequantized operands to FP16 before MMA.
        # Accumulate FP32 here to keep quantization and kernel error separate.
        output = F.linear(x.float(), self.weight.float(), None)
        output = output.to(x.dtype)
        return output if self.bias is None else output + self.bias.to(x.dtype)


def make_quantized_oracle(model):
    oracle = copy.deepcopy(model)
    replacements = [
        (name, DequantizedFP16Oracle(module))
        for name, module in oracle.named_modules()
        if hasattr(module, "packed_weight") and callable(getattr(module, "dequantize", None))
    ]
    if not replacements:
        raise RuntimeError("no quantized modules found; refusing an empty oracle check")
    for name, replacement in replacements:
        parent, child = name.rsplit(".", 1)
        setattr(oracle.get_submodule(parent), child, replacement)
    return oracle.eval(), len(replacements)


@torch.inference_mode()
def trace(model, prompt_ids, decode_steps, forced_tokens=None, *, block_size=16):
    """Return last-position logits and tokens for prefill + fixed decode steps."""
    tokens = len(prompt_ids) + decode_steps
    num_blocks = (tokens + block_size - 1) // block_size
    cache = model.create_cache(num_blocks, block_size)
    blocks = list(range(num_blocks))
    device = cache.device
    logits = model(torch.tensor(prompt_ids, device=device, dtype=torch.long), cache, blocks)[-1:]
    outputs, chosen = [], []
    for step in range(decode_steps + 1):
        outputs.append(logits[0].float().cpu())
        chosen.append(int(logits.argmax(-1).item()))
        if step < decode_steps:
            token = chosen[-1] if forced_tokens is None else forced_tokens[step]
            logits = model(
                torch.tensor([token], device=device, dtype=torch.long), cache, blocks,
                position=len(prompt_ids) + step,
            )
    return {"logits": torch.stack(outputs), "token_ids": chosen}


def compare(reference, candidate, atol, rtol):
    expected, actual = reference["logits"], candidate["logits"]
    error = (actual - expected).abs()
    finite = bool(torch.isfinite(actual).all())
    return {
        "finite": finite,
        "max_abs_logit_error": float(error.max()),
        "mean_abs_logit_error": float(error.mean()),
        "relative_l2_logit_error": float(torch.linalg.vector_norm(error) / torch.linalg.vector_norm(expected).clamp_min(1e-12)),
        "logits_close": bool(torch.allclose(actual, expected, atol=atol, rtol=rtol)),
        "teacher_forced_argmax_match": reference["token_ids"] == candidate["token_ids"],
        "reference_token_ids": reference["token_ids"],
        "candidate_token_ids": candidate["token_ids"],
    }


def accepted(check):
    return check["finite"] and check["logits_close"] and check["greedy_token_match"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--prompt", action="append", dest="prompts")
    parser.add_argument("--decode-steps", type=int, default=8)
    parser.add_argument("--quant-group-size", type=int, choices=[64, 128], default=128)
    parser.add_argument("--linear-backend", choices=["dandan", "dandan_cuda"], default="dandan")
    parser.add_argument("--prefill-backend", choices=["cuda", "dandan"], default="cuda")
    parser.add_argument("--fuse-projections", action="store_true")
    parser.add_argument("--atol", type=float, default=0.125)
    parser.add_argument("--rtol", type=float, default=0.01)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "benchmarks" / "results" / "dandan_qwen3_validation.json")
    args = parser.parse_args()
    configure_local_cache()
    if not Path(args.model).is_dir():
        raise SystemExit("a complete local Qwen3 checkpoint is required")
    if args.decode_steps < 0 or args.atol < 0 or args.rtol < 0:
        raise SystemExit("decode-steps, atol and rtol must be nonnegative")
    prompts = args.prompts or ["用一句话解释 KV Cache。", "Explain continuous batching in one sentence."]
    common = dict(
        device=args.device, dtype=torch.float16, prefill_backend="cuda",
        decode_backend="cuda", elementwise_backend="cuda", data_backend="cuda",
        num_blocks=64, block_size=16, local_files_only=True,
        fuse_projections=args.fuse_projections,
    )
    reference = NativeQwen3Runner.from_pretrained(args.model, linear_backend="cuda", **common)
    prompt_ids = [
        reference.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False,
        ) for prompt in prompts
    ]
    baseline = [trace(reference.model, ids, args.decode_steps) for ids in prompt_ids]
    del reference
    gc.collect()
    torch.cuda.empty_cache()
    candidate = NativeQwen3Runner.from_pretrained(
        args.model, linear_backend=args.linear_backend,
        **{**common, "prefill_backend": args.prefill_backend},
    )
    fp16_checks = []
    for prompt, ids, expected in zip(prompts, prompt_ids, baseline):
        actual = trace(candidate.model, ids, args.decode_steps, expected["token_ids"])
        check = compare(expected, actual, args.atol, args.rtol)
        greedy = trace(candidate.model, ids, args.decode_steps)
        check.update(prompt=prompt, greedy_token_match=expected["token_ids"] == greedy["token_ids"], greedy_token_ids=greedy["token_ids"])
        fp16_checks.append(check)
    fp16_backend = candidate.model.backend_report() if hasattr(candidate.model, "backend_report") else None
    candidate.model.quantize(QuantConfig("int4", "fp16", args.quant_group_size), backend="dandan")
    oracle, replaced = make_quantized_oracle(candidate.model)
    w4_checks, quantization_checks = [], []
    for prompt, ids, base in zip(prompts, prompt_ids, baseline):
        expected = trace(oracle, ids, args.decode_steps)
        actual = trace(candidate.model, ids, args.decode_steps, expected["token_ids"])
        check = compare(expected, actual, args.atol, args.rtol)
        greedy = trace(candidate.model, ids, args.decode_steps)
        check.update(prompt=prompt, greedy_token_match=expected["token_ids"] == greedy["token_ids"], greedy_token_ids=greedy["token_ids"])
        w4_checks.append(check)
        # This comparison deliberately holds the original baseline's input path
        # fixed and is reported only; quantization is allowed to alter logits.
        quantized = trace(candidate.model, ids, args.decode_steps, base["token_ids"])
        quant_error = compare(base, quantized, args.atol, args.rtol)
        quant_error["prompt"] = prompt
        quantization_checks.append(quant_error)
    payload = {
        "status": "passed" if all(accepted(check) for check in fp16_checks + w4_checks) else "failed",
        "model": args.model, "gpu": torch.cuda.get_device_name(torch.device(args.device)),
        "pytorch": torch.__version__, "cuda_runtime": torch.version.cuda,
        "dtype": "fp16", "decode_steps": args.decode_steps,
        "projection_fusion": args.fuse_projections,
        "candidate_prefill_backend": args.prefill_backend,
        "tolerances": {"atol": args.atol, "rtol": args.rtol},
        "fp16_backend": fp16_backend,
        "quantized_backend": candidate.model.backend_report() if hasattr(candidate.model, "backend_report") else None,
        "quantization_report": candidate.model.quantization_report,
        "oracle_quantized_module_count": replaced,
        "fp16_dandan_vs_cublas": fp16_checks,
        "w4a16_kernel_vs_same_weights_oracle": w4_checks,
        "quantization_change_vs_original_fp16": quantization_checks,
        "note": "Fixed short greedy traces ignore EOS. W4 oracle uses the identical packed state's FP16-rounded dequantized weights and FP32 accumulation. Quantization change is descriptive, not a correctness acceptance test. CUDA Graph and concurrent request lifecycle require separate tests.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    raise SystemExit(0 if payload["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
