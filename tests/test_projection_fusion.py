import pytest
import torch

from mini_engine.kernels.linear import FusedLinear
from mini_engine.model import NativeQwen3, Qwen3Config


def test_fused_linear_matches_concatenated_linears():
    torch.manual_seed(81)
    first = torch.nn.Linear(11, 7, bias=False)
    second = torch.nn.Linear(11, 5, bias=False)
    fused = FusedLinear.from_linears((first, second))
    x = torch.randn(3, 2, 11)
    expected = torch.cat((first(x), second(x)), dim=-1)
    torch.testing.assert_close(fused(x), expected)


def test_native_projection_fusion_matches_prefill_and_decode():
    torch.manual_seed(82)
    config = Qwen3Config(
        vocab_size=53,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
    )
    reference = NativeQwen3(config, prefill_backend="math")
    fused = NativeQwen3(config, prefill_backend="math")
    fused.load_state_dict(reference.state_dict(), strict=True)
    assert fused.enable_projection_fusion() == 4
    assert fused.projection_fusion_enabled
    assert all(layer.self_attn.projection_fused for layer in fused.model.layers)
    assert all(layer.mlp.projection_fused for layer in fused.model.layers)

    tokens = torch.tensor([3, 8, 11, 5, 9], dtype=torch.long)
    ref_cache = reference.create_cache(8, 4)
    fused_cache = fused.create_cache(8, 4)
    blocks = [6, 1, 4, 0]
    with torch.inference_mode():
        expected = reference(tokens, ref_cache, blocks)
        actual = fused(tokens, fused_cache, blocks)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
        for position in range(tokens.numel(), tokens.numel() + 4):
            token = expected[-1].argmax().view(1)
            expected = reference(token, ref_cache, blocks, position=position)
            actual = fused(token, fused_cache, blocks, position=position)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


def test_projection_fusion_rejects_quantized_model():
    config = Qwen3Config(17, 16, 32, 1, 2, 1, 8, max_position_embeddings=32)
    model = NativeQwen3(config, prefill_backend="math")
    from mini_engine.config import QuantConfig

    model.quantize(QuantConfig("int8", activation_dtype="fp32", group_size=8))
    with pytest.raises(ValueError, match="before quantization"):
        model.enable_projection_fusion()


def test_quantization_rejects_fused_model():
    config = Qwen3Config(17, 16, 32, 1, 2, 1, 8, max_position_embeddings=32)
    model = NativeQwen3(config, prefill_backend="math")
    model.enable_projection_fusion()
    from mini_engine.config import QuantConfig

    with pytest.raises(ValueError, match="cannot be combined"):
        model.quantize(QuantConfig("int8", activation_dtype="fp32", group_size=8))
