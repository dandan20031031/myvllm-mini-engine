"""Real pack/dequant/model checks; no external weights are downloaded."""

from dataclasses import asdict

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mini_engine.config import QuantConfig
from mini_engine.model import NativeQwen3, Qwen3Config
from mini_engine.quantization import ReferenceQuantLinear, create_backend
from mini_engine.quantization.capabilities import precision_capabilities

FORMATS = ["int8", "int4", "fp8_e4m3", "fp8_e5m2"]


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_roundtrip_and_linear(format, device):
    torch.manual_seed(5)
    backend = create_backend(QuantConfig(format, activation_dtype="fp32", group_size=8))
    weight = torch.randn(5, 19, device=device)
    weight[0] = 0
    packed = backend.pack(weight)
    restored = backend.dequantize(packed)
    assert restored.shape == weight.shape and restored.dtype == torch.float32
    assert torch.isfinite(restored).all() and not restored[0].any()
    relative_rmse = ((restored - weight).square().mean() / weight.square().mean()).sqrt()
    limits = {"int8": 0.015, "int4": 0.15, "fp8_e4m3": 0.06, "fp8_e5m2": 0.12}
    assert relative_rmse < limits[format]
    x = torch.randn(3, 19, device=device)
    bias = torch.randn(5, device=device)
    torch.testing.assert_close(backend.linear(x, packed, bias), F.linear(x, restored, bias))
    assert packed.scale.dtype == torch.float32


def test_int4_nibble_order_and_padding():
    backend = create_backend(QuantConfig("int4", group_size=4))
    weight = torch.tensor([[-7.0, -1.0, 0.0, 7.0, 2.0]])
    packed = backend.pack(weight)
    assert packed.weight.dtype == torch.uint8
    assert packed.weight.shape == (1, 4)
    assert packed.weight[0, :2].tolist() == [0xF9, 0x70]
    torch.testing.assert_close(backend.dequantize(packed), weight)
    assert packed.in_features == 5
    # At realistic shape the packed payload plus scales is smaller than BF16.
    matrix = torch.randn(16, 256, dtype=torch.bfloat16)
    larger = create_backend(QuantConfig("int4")).pack(matrix)
    assert larger.weight.numel() + larger.scale.numel() * 4 < matrix.numel() * 2


@pytest.mark.parametrize("format", FORMATS)
def test_tiny_and_extreme_weights_do_not_overflow(format):
    backend = create_backend(QuantConfig(format, group_size=4))
    values = torch.tensor([[0.0, 1e-30, -1e-30, 1e-29], [1e30, -1e30, 0.0, 1.0]])
    packed = backend.pack(values)
    restored = backend.dequantize(packed)
    assert (packed.scale > 0).all() and torch.isfinite(restored).all()


@pytest.mark.parametrize("format", FORMATS)
def test_quantized_module_buffers_state_and_dtype_movement(format):
    torch.manual_seed(9)
    config = QuantConfig(format, activation_dtype="fp32", group_size=8)
    module = ReferenceQuantLinear(nn.Linear(19, 7), config)
    clone = ReferenceQuantLinear(nn.Linear(19, 7), config)
    clone.load_state_dict(module.state_dict())
    x = torch.randn(3, 19)
    torch.testing.assert_close(module(x), clone(x))
    assert not list(module.parameters())  # no original floating matrix retained
    scale = module.scales.clone()
    module.to(dtype=torch.bfloat16)
    assert module.scales.dtype == torch.float32
    assert torch.equal(module.scales, scale)
    assert module.packed_weight.dtype in (torch.uint8, torch.int8)


@pytest.mark.parametrize("format", FORMATS)
def test_native_projection_replacement_and_cached_decode(format):
    torch.manual_seed(11)
    config = Qwen3Config(37, 32, 64, 2, 4, 2, 16, max_position_embeddings=64)
    model = NativeQwen3(config, "math")
    embedding = model.model.embed_tokens.weight
    norm = model.model.norm.weight
    report = model.quantize(QuantConfig(format, activation_dtype="fp32", group_size=8))
    assert len(report["replaced_layers"]) == 14 and report["reference_only"]
    assert not report["accelerated_quant_gemm"]
    assert model.model.embed_tokens.weight is embedding and model.lm_head.weight is embedding
    assert model.model.norm.weight is norm
    # Compare cached execution to recomputation with exactly the same quantized weights.
    a_cache, b_cache = model.create_cache(8, 4), model.create_cache(8, 4)
    blocks = [5, 1, 6, 0]
    ids = torch.tensor([1, 2, 3, 4, 5])
    a = model(ids, a_cache, blocks)
    for _ in range(3):
        token = a[-1].argmax().view(1)
        position = ids.numel()
        ids = torch.cat([ids, token])
        a = model(token, a_cache, blocks, position)
        b = model(ids, b_cache, blocks)
        torch.testing.assert_close(a[-1], b[-1], atol=2e-5, rtol=2e-5)


def test_configuration_and_unimplemented_formats_fail_before_mutation():
    with pytest.raises(ValueError):
        QuantConfig("tf32")
    with pytest.raises(NotImplementedError):
        QuantConfig("int8", symmetric=False)
    with pytest.raises(ValueError):
        QuantConfig("int4", group_size=3)
    assert QuantConfig("fp8").canonical_weight_dtype == "fp8_e4m3"
    for format in ("nvfp4", "mxfp8"):
        with pytest.raises(NotImplementedError, match="reserved"):
            create_backend(QuantConfig(format))
    backend = create_backend(QuantConfig("int8"))
    with pytest.raises(ValueError, match="finite"):
        backend.pack(torch.tensor([[float("nan")]]))
    with pytest.raises(ValueError, match="nonempty"):
        backend.pack(torch.empty(0, 3))
    model = NativeQwen3(Qwen3Config(17, 16, 32, 1, 2, 1, 8))
    with pytest.raises(NotImplementedError):
        model.quantize(QuantConfig("nvfp4"))
    assert isinstance(model.model.layers[0].self_attn.q_proj, nn.Linear)
    model.quantize(QuantConfig("int8"))
    with pytest.raises(ValueError, match="already"):
        model.quantize(QuantConfig("int4"))


def test_capabilities_do_not_claim_fp4_acceleration():
    before = torch.backends.cuda.matmul.allow_tf32
    report = precision_capabilities("cpu", probe=True)
    assert torch.backends.cuda.matmul.allow_tf32 == before
    assert report["formats"]["fp8_e4m3"]["tensor_conversion_verified"] is True
    assert report["formats"]["nvfp4"]["status"] == "reserved"
    assert not report["formats"]["nvfp4"]["accelerated_quant_gemm_implemented"]
    assert not report["tf32"]["weight_storage_format"]


@pytest.mark.parametrize("format", FORMATS)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_gpu_quantized_activation_dtype_and_model_generation(format, dtype):
    from mini_engine.model import NativeQwen3Runner
    from test_native_qwen3 import TinyTokenizer
    torch.manual_seed(3)
    model = NativeQwen3(Qwen3Config(37, 32, 64, 1, 4, 2, 16), "math").to(device="cuda", dtype=dtype)
    activation = "fp16" if dtype == torch.float16 else "bf16"
    model.quantize(QuantConfig(format, activation_dtype=activation, group_size=8))
    runner = NativeQwen3Runner(model, TinyTokenizer(), num_blocks=8, block_size=4)
    result = runner.generate_incremental("hello", max_new_tokens=4)
    assert len(result.token_ids) == 4 and result.finish_reason == "length"
    assert runner.block_manager.free_blocks == 8
    assert all(torch.isfinite(module.scales).all() for module in model.modules() if isinstance(module, ReferenceQuantLinear))
