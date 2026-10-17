"""Native-model Linear replacement backed by packed reference weights."""

import torch
from torch import nn

from .backends import FLOAT_DTYPES, PackedFP8, PackedInt4, PackedInt8, create_backend


class ReferenceQuantLinear(nn.Module):
    """Inference-only module. Original floating weights are not retained.

    Byte payloads stay integer buffers, including FP8 bytes, so module.to(dtype)
    cannot accidentally expand FP8 storage. Scales always remain FP32.
    """

    def __init__(self, linear, config):
        super().__init__()
        self.backend = create_backend(config)
        if config.canonical_weight_dtype not in {"int8", "int4", "fp8_e4m3", "fp8_e5m2"}:
            raise ValueError("ReferenceQuantLinear requires an INT8/INT4/FP8 weight format")
        self.quant_config = config
        self.format = config.canonical_weight_dtype
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        packed = self.backend.pack(linear.weight)
        data = packed.weight.view(torch.uint8) if self.format.startswith("fp8") else packed.weight
        self.register_buffer("packed_weight", data)
        self.register_buffer("scales", packed.scale)
        self.register_buffer("bias", None if linear.bias is None else linear.bias.detach().clone())
        self.eval()

    def _apply(self, fn, recurse=True):
        scales = self.scales
        super()._apply(fn, recurse=recurse)
        self.scales = scales.to(device=self.packed_weight.device, dtype=torch.float32)
        return self

    def packed(self):
        if self.format == "int8":
            return PackedInt8(self.packed_weight, self.scales)
        if self.format == "int4":
            return PackedInt4(self.packed_weight, self.scales, self.in_features, self.quant_config.group_size)
        return PackedFP8(self.packed_weight.view(self.backend.dtype), self.scales)

    @property
    def storage_bytes(self):
        return sum(t.numel() * t.element_size() for t in (self.packed_weight, self.scales))

    def dequantize(self):
        return self.backend.dequantize(self.packed())

    def forward(self, x):
        activation = x.to(FLOAT_DTYPES[self.quant_config.activation_dtype])
        return self.backend.linear(activation, self.packed(), self.bias).to(x.dtype)


def quantize_model_linears(model, config):
    """Replace the seven projections in each decoder layer; keep LM head high precision.

    Prepare replacements before mutating the model, so a pack failure cannot
    leave the model partially quantized. Caller must quantize after loading
    and moving the floating model to its target device.
    """
    backend = create_backend(config)  # Fail on reserved formats before mutation.
    if not backend.reference_only:
        raise ValueError("floating models should use model.to(dtype), not reference quantization")
    if getattr(model, "quantization_report", None) is not None:
        raise ValueError("model is already quantized; reload floating weights to change format")
    selected = [(name, module) for name, module in model.named_modules()
                if name.startswith("model.layers.") and isinstance(module, nn.Linear)]
    if not selected:
        raise ValueError("no native Qwen3 decoder Linear modules found")
    replacements = [(name, ReferenceQuantLinear(module, config)) for name, module in selected]
    original_bytes = sum(module.weight.numel() * module.weight.element_size() for _, module in selected)
    packed_bytes = sum(module.storage_bytes for _, module in replacements)
    for name, replacement in replacements:
        parent_name, child_name = name.rsplit(".", 1)
        setattr(model.get_submodule(parent_name), child_name, replacement)
    report = {
        "weight_format": config.canonical_weight_dtype,
        "activation_dtype": config.activation_dtype,
        "compute_backend": "dequantize_fp32_linear_reference",
        "reference_only": True,
        "accelerated_quant_gemm": False,
        "replaced_layers": [name for name, _ in replacements],
        "original_projection_weight_bytes": original_bytes,
        "packed_projection_weight_and_scale_bytes": packed_bytes,
        "projection_storage_ratio": original_bytes / packed_bytes,
        "high_precision_modules": ["embedding", "all_norms", "residual", "lm_head", "kv_cache"],
    }
    model.quantization_report = report
    return report
