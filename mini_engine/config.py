from dataclasses import dataclass
from typing import ClassVar, Literal


WeightDType = Literal[
    "fp32", "bf16", "fp16", "int8", "int4", "fp8", "fp8_e4m3", "fp8_e5m2",
    "mxfp8", "nvfp4",
]


@dataclass(frozen=True)
class QuantConfig:
    """Weight storage format; activation precision is independent.

    fp8 aliases fp8_e4m3. MXFP8/NVFP4 are reserved, explicitly unavailable.
    Reference low-bit paths dequantize into FP32. The optional dandan compute
    backend supports INT4 weights with FP16 activations (group size 64/128).
    """

    weight_dtype: WeightDType = "bf16"
    activation_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    group_size: int = 128
    symmetric: bool = True

    def __post_init__(self):
        if self.weight_dtype not in {
            "fp32", "bf16", "fp16", "int8", "int4", "fp8", "fp8_e4m3",
            "fp8_e5m2", "mxfp8", "nvfp4",
        }:
            raise ValueError("unknown weight format; TF32 is a GEMM policy, not weight storage")
        if self.activation_dtype not in {"fp32", "fp16", "bf16"}:
            raise ValueError("activation_dtype must be fp32, fp16 or bf16")
        if not isinstance(self.group_size, int) or isinstance(self.group_size, bool) or self.group_size <= 0:
            raise ValueError("group_size must be a positive integer")
        if self.weight_dtype == "int4" and self.group_size % 2:
            raise ValueError("INT4 group_size must be even for two nibbles per byte")
        if self.symmetric is not True:
            raise NotImplementedError("only symmetric quantization is implemented")

    @property
    def canonical_weight_dtype(self) -> str:
        return "fp8_e4m3" if self.weight_dtype == "fp8" else self.weight_dtype


@dataclass(frozen=True)
class RunnerConfig:
    """Native model startup settings shared by chat and HTTP entry points.

    This validates static combinations without loading weights or compiling
    kernels. Hardware and operator-library checks remain with the runner.
    Scheduler/request limits belong to the application using that runner.
    """

    BACKENDS: ClassVar[dict[str, tuple[str, ...]]] = {
        "prefill": ("auto", "flash", "math", "cuda", "dandan"),
        "decode": ("torch", "cuda"),
        "elementwise": ("torch", "cuda"),
        "data": ("torch", "cuda"),
        "linear": ("torch", "cuda", "dandan", "dandan_cuda"),
        "sampling": ("torch", "cuda"),
        "quant": ("reference", "dandan"),
    }

    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda"
    dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    num_blocks: int = 256
    block_size: int = 16
    prefill_backend: str = "auto"
    decode_backend: str = "torch"
    elementwise_backend: str = "torch"
    data_backend: str = "torch"
    linear_backend: str = "torch"
    sampling_backend: str = "torch"
    quant_backend: str = "reference"
    quantization: QuantConfig | None = None
    fuse_projections: bool = False
    local_files_only: bool = False
    enable_thinking: bool = False
    use_cuda_graph: bool = False

    def __post_init__(self):
        if not self.model_name_or_path or not self.device:
            raise ValueError("model and device must be nonempty")
        if self.dtype not in {"bf16", "fp16", "fp32"}:
            raise ValueError("dtype must be bf16, fp16 or fp32")
        for name in ("num_blocks", "block_size"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, choices in self.BACKENDS.items():
            if getattr(self, name + "_backend") not in choices:
                raise ValueError(f"unknown {name} backend")
        cuda = self.device.split(":", 1)[0] == "cuda"
        low_precision = self.dtype in {"fp16", "bf16"}
        for name in ("prefill", "decode", "elementwise", "data"):
            if getattr(self, name + "_backend") == "cuda" and not (cuda and low_precision):
                raise ValueError(f"CUDA {name} requires a CUDA device and FP16/BF16")
        if (self.linear_backend == "cuda" or self.sampling_backend == "cuda") and not cuda:
            raise ValueError("CUDA linear/sampling backends require a CUDA device")
        if self.use_cuda_graph and not (
            cuda and low_precision and self.decode_backend == "cuda" and self.data_backend == "cuda"
        ):
            raise ValueError("--cuda-graph requires CUDA FP16/BF16, --decode-backend cuda and --data-backend cuda")
        if self.quantization is not None:
            if self.quantization.activation_dtype != self.dtype:
                raise ValueError("quantization activation dtype must match runner dtype")
            if self.quant_backend == "reference" and self.fuse_projections:
                raise ValueError("reference quantization and projection fusion cannot be combined yet")
        if self.quant_backend == "dandan":
            if (self.quantization is None or self.quantization.canonical_weight_dtype != "int4"
                    or self.quantization.group_size not in (64, 128)):
                raise ValueError("dandan quant backend requires INT4 with group size 64 or 128")
        if (self.linear_backend in {"dandan", "dandan_cuda"}
                or self.prefill_backend == "dandan" or self.quant_backend == "dandan"):
            if not cuda or self.dtype != "fp16":
                raise ValueError("dandan backend requires a CUDA device and FP16 weights/activations")

    def runner_kwargs(self) -> dict:
        """Translate settings once; preserve the runner's public load API."""
        import torch

        return {
            "device": self.device,
            "dtype": {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[self.dtype],
            **{name + "_backend": getattr(self, name + "_backend") for name in self.BACKENDS},
            "num_blocks": self.num_blocks,
            "block_size": self.block_size,
            "enable_thinking": self.enable_thinking,
            "local_files_only": self.local_files_only,
            "fuse_projections": self.fuse_projections,
            "quantization": self.quantization,
            "use_cuda_graph": self.use_cuda_graph,
        }


@dataclass(frozen=True)
class EngineConfig:
    """Foundation facade/scheduler settings; native applications use RunnerConfig."""

    model_name_or_path: str = "Qwen/Qwen3-0.6B"
    device: str = "cuda"
    dtype: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    block_size: int = 16
    num_gpu_blocks: int = 256
    max_num_batched_tokens: int = 4096
    max_new_tokens: int = 128
    quantization: QuantConfig = QuantConfig()
    enable_thinking: bool = False
    use_custom_kernels: bool = False

    def torch_dtype_name(self) -> str:
        if self.dtype != "auto":
            return self.dtype
        return "bf16" if self.device.startswith("cuda") else "fp32"
