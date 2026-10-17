from dataclasses import dataclass
from typing import Literal


WeightDType = Literal["bf16", "fp16", "int8", "int4", "fp8"]


@dataclass(frozen=True)
class QuantConfig:
    """Configuration shared by all quantization backends.

    The first implementation only provides floating-point and a reference
    W8A16 path.  The interface intentionally leaves room for packed W4A16 and
    FP8 kernels without changing model code.
    """

    weight_dtype: WeightDType = "bf16"
    activation_dtype: Literal["bf16", "fp16", "fp32"] = "bf16"
    group_size: int = 128
    symmetric: bool = True


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
