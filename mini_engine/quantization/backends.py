from dataclasses import dataclass

import torch

from ..config import QuantConfig


class QuantBackend:
    def pack(self, weight: torch.Tensor):
        raise NotImplementedError

    def linear(self, x: torch.Tensor, packed_weight, bias: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError


class FloatBackend(QuantBackend):
    def __init__(self, dtype: torch.dtype = torch.bfloat16):
        self.dtype = dtype

    def pack(self, weight: torch.Tensor):
        return weight.to(self.dtype)

    def linear(self, x, packed_weight, bias=None):
        return torch.nn.functional.linear(x.to(packed_weight.dtype), packed_weight, bias)


@dataclass
class PackedInt8:
    weight: torch.Tensor
    scale: torch.Tensor


class Int8WeightOnlyBackend(QuantBackend):
    """Reference per-output-channel W8A16 implementation."""

    def pack(self, weight: torch.Tensor) -> PackedInt8:
        scale = weight.abs().amax(dim=1, keepdim=True).clamp_min(1e-8) / 127.0
        packed = torch.round(weight / scale).to(torch.int8)
        return PackedInt8(packed, scale.squeeze(1).to(weight.dtype))

    def linear(self, x, packed_weight: PackedInt8, bias=None):
        weight = packed_weight.weight.float() * packed_weight.scale.float().unsqueeze(1)
        return torch.nn.functional.linear(x.float(), weight, bias).to(x.dtype)


def create_backend(config: QuantConfig) -> QuantBackend:
    if config.weight_dtype in ("bf16", "fp16", "fp32"):
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[config.weight_dtype]
        return FloatBackend(dtype)
    if config.weight_dtype == "int8":
        return Int8WeightOnlyBackend()
    raise NotImplementedError(f"quantization backend not implemented yet: {config.weight_dtype}")
