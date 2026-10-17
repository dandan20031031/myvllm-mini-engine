"""Inference-only, correctness-first packed weight backends.

No INT8/INT4/FP8 Tensor Core GEMM is claimed: each quantized Linear dequantizes
to FP32, performs FP32 Linear, then casts back to the input activation dtype.
"""

from dataclasses import dataclass

import torch
from torch.nn import functional as F

from ..config import QuantConfig

FLOAT_DTYPES = {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}


def _weight_fp32(weight):
    if weight.ndim != 2 or not weight.is_floating_point() or not all(weight.shape):
        raise ValueError("weight must be a nonempty floating-point [out_features, in_features] matrix")
    data = weight.detach().float()
    if not bool(torch.isfinite(data).all()):
        raise ValueError("weight must contain only finite values")
    return data


def _scale(maximum, limit):
    # FP32 scales prevent tiny FP16 weights from producing zero/infinite scales.
    return torch.where(
        maximum == 0, torch.ones_like(maximum),
        (maximum / limit).clamp_min(torch.finfo(torch.float32).tiny),
    )


class QuantBackend:
    reference_only = True

    def pack(self, weight):
        raise NotImplementedError

    def dequantize(self, packed_weight):
        raise NotImplementedError

    def linear(self, x, packed_weight, bias=None):
        weight = self.dequantize(packed_weight)
        if x.device != weight.device:
            raise ValueError("activations and packed weights must be on the same device")
        return F.linear(x.float(), weight.float(), None if bias is None else bias.float()).to(x.dtype)


class FloatBackend(QuantBackend):
    reference_only = False

    def __init__(self, dtype=torch.bfloat16):
        self.dtype = dtype

    def pack(self, weight):
        _weight_fp32(weight)
        return weight.detach().to(self.dtype).contiguous()

    def dequantize(self, packed_weight):
        return packed_weight.float()

    def linear(self, x, packed_weight, bias=None):
        return F.linear(x.to(packed_weight.dtype), packed_weight,
                        None if bias is None else bias.to(packed_weight.dtype))


@dataclass
class PackedInt8:
    weight: torch.Tensor
    scale: torch.Tensor


class Int8WeightOnlyBackend(QuantBackend):
    """Symmetric signed INT8, one FP32 scale per output channel."""

    def pack(self, weight):
        data = _weight_fp32(weight)
        scale = _scale(data.abs().amax(dim=1, keepdim=True), 127.0)
        packed = torch.round(data / scale).clamp(-127, 127).to(torch.int8)
        return PackedInt8(packed.contiguous(), scale.squeeze(1).contiguous())

    def dequantize(self, packed_weight):
        return packed_weight.weight.float() * packed_weight.scale.float().unsqueeze(1)


@dataclass
class PackedInt4:
    weight: torch.Tensor
    scale: torch.Tensor
    in_features: int
    group_size: int


class Int4WeightOnlyBackend(QuantBackend):
    """Symmetric [-7,7], two signed two's-complement nibbles per uint8.

    Per-row groups are padded with zero to group_size; padding is not an
    extra logical input feature. Low nibble is the earlier input column.
    """

    def __init__(self, group_size=128):
        if not isinstance(group_size, int) or isinstance(group_size, bool) or group_size <= 0 or group_size % 2:
            raise ValueError("INT4 group_size must be a positive even integer")
        self.group_size = group_size

    def pack(self, weight):
        data = _weight_fp32(weight)
        out, features = data.shape
        groups = (features + self.group_size - 1) // self.group_size
        grouped = F.pad(data, (0, groups * self.group_size - features)).reshape(out, groups, self.group_size)
        scale = _scale(grouped.abs().amax(dim=-1), 7.0)
        quantized = torch.round(grouped / scale.unsqueeze(-1)).clamp(-7, 7).to(torch.int16)
        nibbles = (quantized.reshape(out, -1) & 15).to(torch.uint8)
        packed = nibbles[:, 0::2] | (nibbles[:, 1::2] << 4)
        return PackedInt4(packed.contiguous(), scale.contiguous(), features, self.group_size)

    def dequantize(self, packed_weight):
        low = (packed_weight.weight & 15).to(torch.int16)
        high = (packed_weight.weight >> 4).to(torch.int16)
        nibbles = torch.stack((low, high), dim=-1).flatten(1)
        signed = torch.where(nibbles >= 8, nibbles - 16, nibbles).float()
        out, groups = packed_weight.scale.shape
        data = signed.reshape(out, groups, packed_weight.group_size) * packed_weight.scale.float().unsqueeze(-1)
        return data.reshape(out, -1)[:, :packed_weight.in_features].contiguous()


@dataclass
class PackedFP8:
    weight: torch.Tensor
    scale: torch.Tensor


class FP8WeightOnlyBackend(QuantBackend):
    """Per-output-channel scaled E4M3FN or E5M2 storage, float activations.

    This is W8A16/W8A32 reference computation, NOT FP8 W8A8 GEMM.
    """

    def __init__(self, format="fp8_e4m3"):
        if format not in {"fp8", "fp8_e4m3", "fp8_e5m2"}:
            raise ValueError("unknown FP8 format")
        self.format = "fp8_e4m3" if format == "fp8" else format
        attr = "float8_e4m3fn" if self.format == "fp8_e4m3" else "float8_e5m2"
        self.dtype = getattr(torch, attr, None)
        if self.dtype is None:
            raise NotImplementedError(f"this PyTorch version has no {attr}")

    def pack(self, weight):
        data = _weight_fp32(weight)
        maximum = torch.finfo(self.dtype).max
        scale = _scale(data.abs().amax(dim=1, keepdim=True), maximum)
        # Clamp explicitly: overflow must not turn a finite weight into NaN/Inf.
        packed = (data / scale).clamp(-maximum, maximum).to(self.dtype)
        return PackedFP8(packed.contiguous(), scale.squeeze(1).contiguous())

    def dequantize(self, packed_weight):
        return packed_weight.weight.float() * packed_weight.scale.float().unsqueeze(1)


def create_backend(config):
    format = config.canonical_weight_dtype
    if format in FLOAT_DTYPES:
        return FloatBackend(FLOAT_DTYPES[format])
    if format == "int8":
        return Int8WeightOnlyBackend()
    if format == "int4":
        return Int4WeightOnlyBackend(config.group_size)
    if format in {"fp8_e4m3", "fp8_e5m2"}:
        return FP8WeightOnlyBackend(format)
    raise NotImplementedError(
        f"{format} is reserved: block scaling and a Windows CUDA compute backend "
        "are not implemented; INT4/ordinary FP8 are not NVFP4/MXFP8"
    )
