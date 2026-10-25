"""Shared native startup arguments; application-specific options stay at entry points."""

from .config import QuantConfig, RunnerConfig


def add_runner_arguments(parser, *, serving=False):
    """Register one source of model options, retaining the HTTP subset."""
    group = parser.add_argument_group("model and execution")
    if serving:
        group.add_argument("--model", required=True)
    else:
        group.add_argument("--model", default="Qwen/Qwen3-0.6B")
    group.add_argument("--device", default="cuda")
    group.add_argument("--dtype", choices=["bf16", "fp16", "fp32"], default="bf16")
    group.add_argument("--num-blocks", type=int, default=256)
    group.add_argument("--block-size", type=int, default=16)
    for name, choices in RunnerConfig.BACKENDS.items():
        if serving and name == "prefill":
            choices = tuple(choice for choice in choices if choice != "flash")
        default = "auto" if name == "prefill" else "reference" if name == "quant" else "torch"
        group.add_argument(f"--{name}-backend", choices=choices, default=default)
    formats = ("none", "int4") if serving else (
        "none", "int8", "int4", "fp8", "fp8_e4m3", "fp8_e5m2", "mxfp8", "nvfp4")
    group.add_argument("--quantization", choices=formats, default="none")
    group.add_argument("--quant-group-size", type=int, default=128, **({"choices": (64, 128)} if serving else {}))
    group.add_argument("--fuse-projections", action="store_true",
                       help="fuse QKV and Gate/Up projections into concatenated matmuls")
    group.add_argument("--cuda-graph", action="store_true",
                       help="capture decode and greedy sampling; requires CUDA decode/data backends")
    group.add_argument("--local-files-only", action="store_true")
    if not serving:
        group.add_argument("--thinking", action="store_true")


def runner_config_from_args(args) -> RunnerConfig:
    quantization = None if args.quantization == "none" else QuantConfig(
        args.quantization, activation_dtype=args.dtype, group_size=args.quant_group_size)
    return RunnerConfig(
        model_name_or_path=args.model, device=args.device, dtype=args.dtype,
        num_blocks=args.num_blocks, block_size=args.block_size,
        **{name + "_backend": getattr(args, name + "_backend") for name in RunnerConfig.BACKENDS},
        quantization=quantization, fuse_projections=args.fuse_projections,
        local_files_only=args.local_files_only, enable_thinking=getattr(args, "thinking", False),
        use_cuda_graph=args.cuda_graph,
    )
