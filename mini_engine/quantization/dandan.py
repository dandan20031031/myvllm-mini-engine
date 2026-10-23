"""Lossless signed INT4 storage bridge to the optional W4A16 compute backend."""

import torch
from torch import nn

from ..kernels.dandan_backend import require_dandan_capability, validate_dandan_runtime
from ..kernels.linear import FusedLinear
from .backends import Int4WeightOnlyBackend


def validate_quant_config(config):
    if (config is None or config.canonical_weight_dtype != "int4"
            or config.activation_dtype != "fp16" or config.group_size not in (64, 128)
            or not config.symmetric):
        raise ValueError("dandan quant backend requires symmetric INT4, FP16 activation, group_size=64 or 128")


class DandanQuantLinear(nn.Module):
    """Quantize once, convert packing once, and keep only packed CUDA weights."""

    def __init__(self, linear, config):
        super().__init__()
        validate_quant_config(config)
        validate_dandan_runtime(linear.weight.device, linear.weight.dtype)
        ops = require_dandan_capability("int4")
        self.in_features, self.out_features = linear.in_features, linear.out_features
        self.quant_config = config
        source = Int4WeightOnlyBackend(config.group_size).pack(linear.weight)
        # XOR 8 maps two's-complement signed nibbles to unsigned q+8 exactly.
        nibbles = torch.stack((source.weight & 15, source.weight >> 4), dim=-1).flatten(1)
        unsigned = (nibbles[:, :self.in_features] ^ 8).contiguous()
        packed = ops.pack_int4(unsigned)
        groups = (self.in_features + 63) // 64
        scales = source.scale.repeat_interleave(config.group_size // 64, dim=1)[:, :groups].contiguous()
        zeros = torch.full_like(scales, 8, dtype=torch.int32)
        self.op = ops.QuantizedLinear(packed, scales, zeros, in_features=self.in_features,
                                      group_size=64, bias=linear.bias, backend="cuda")
        self.eval()

    @property
    def bias(self):
        return self.op.bias

    @property
    def storage_bytes(self):
        return self.op.storage_bytes

    def dequantize(self):
        return self.op.dequantize()

    def get_extra_state(self):
        return {"bridge_version": 1, "source_group_size": self.quant_config.group_size,
                "original_shape": [self.out_features, self.in_features],
                "scheme": "symmetric_signed_to_unsigned_zero8"}

    def set_extra_state(self, state):
        if state != self.get_extra_state():
            raise ValueError("incompatible dandan INT4 bridge metadata/source group size")

    def forward(self, x):
        if x.ndim < 1 or x.shape[-1] != self.in_features:
            raise ValueError("input feature dimension does not match quantized Linear")
        y = self.op(x.reshape(-1, self.in_features))
        return y.reshape(*x.shape[:-1], self.out_features)


def quantize_dandan_linears(model, config):
    validate_quant_config(config)
    validate_dandan_runtime(model.model.embed_tokens.weight.device, model.model.embed_tokens.weight.dtype)
    if model.quantization_report is not None:
        raise ValueError("model is already quantized; reload floating weights")
    selected = [(name, mod) for name, mod in model.named_modules()
                if name.startswith("model.layers.") and isinstance(mod, (nn.Linear, FusedLinear))]
    if not selected:
        raise ValueError("no native Qwen3 decoder Linear modules found")
    # No mutation before every replacement has successfully packed/validated.
    replacements = [(name, DandanQuantLinear(mod, config)) for name, mod in selected]
    original_bytes = sum(mod.weight.numel() * mod.weight.element_size() for _, mod in selected)
    packed_bytes = sum(mod.storage_bytes for _, mod in replacements)
    for name, replacement in replacements:
        parent, child = name.rsplit(".", 1)
        setattr(model.get_submodule(parent), child, replacement)
    report = {
        "weight_format": "int4", "activation_dtype": "fp16", "format_version": 2,
        "scheme": "symmetric_signed_to_unsigned_zero8", "source_group_size": config.group_size,
        "storage_group_size": 64, "fp16_weight_rounding": "fp16_before_multiply",
        "compute_backend": "dandan_w4a16_gemv_or_tensorcore", "reference_only": False,
        "accelerated_quant_gemm": True, "projection_fusion": model.projection_fusion_enabled,
        "replaced_layers": [name for name, _ in replacements],
        "original_projection_weight_bytes": original_bytes,
        "packed_projection_weight_and_scale_bytes": packed_bytes,
        "projection_storage_ratio": original_bytes / packed_bytes,
        "high_precision_modules": ["embedding", "all_norms", "residual", "lm_head", "kv_cache"],
    }
    model.quantization_report = report
    return report
