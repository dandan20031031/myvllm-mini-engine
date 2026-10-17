from dataclasses import dataclass
from typing import Literal


WeightDType = Literal[
    "fp32", "bf16", "fp16", "int8", "int4", "fp8", "fp8_e4m3", "fp8_e5m2",
    "mxfp8", "nvfp4",
]


@dataclass(frozen=True)
class QuantConfig:
    """Weight storage format; activation precision is independent.

    fp8 aliases fp8_e4m3. MXFP8/NVFP4 are reserved, explicitly unavailable.
    The implemented low-bit paths are dequantize + FP32 Linear references.
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
class EngineConfig:
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
