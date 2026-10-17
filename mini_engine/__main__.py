"""PowerShell: python -m mini_engine --model D:\\models\\Qwen3-0.6B --prompt hello"""

import argparse
import json
from dataclasses import asdict

import torch

from .model import NativeQwen3Runner
from .config import QuantConfig


def main():
    parser = argparse.ArgumentParser(description="Single-request native Qwen3 inference")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=256)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--prefill-backend", choices=["auto", "flash", "math"], default="auto")
    parser.add_argument("--decode-backend", choices=["torch", "cuda"], default="torch")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--quantization", choices=["none", "int8", "int4", "fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "nvfp4"], default="none")
    parser.add_argument("--quant-group-size", type=int, default=128)
    args = parser.parse_args()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]
    runner = NativeQwen3Runner.from_pretrained(
        args.model, device=args.device, dtype=dtype, prefill_backend=args.prefill_backend, decode_backend=args.decode_backend,
        num_blocks=args.num_blocks, block_size=args.block_size,
        enable_thinking=args.thinking, local_files_only=args.local_files_only,
        quantization=None if args.quantization == "none" else QuantConfig(
            args.quantization, activation_dtype=args.dtype, group_size=args.quant_group_size),
    )
    result = runner.generate_incremental(args.prompt, args.max_new_tokens)
    payload = asdict(result)
    payload["quantization"] = runner.model.quantization_report
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
