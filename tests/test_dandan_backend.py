"""Optional dandan_ops integration contracts; no model downloads are needed.

CUDA projection tests use the prebuilt operator library. Full-model graph
checks additionally require MINI_ENGINE_TEST_CUDA=1 because they load the
engine's own CUDA extension. Quantized oracles reuse the exact packed state
and FP16 weight rounding, so kernel error is separate from INT4 quantization
error.
"""

import copy
import importlib
import importlib.util
import os
import sys
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from mini_engine.config import QuantConfig
from mini_engine.kernels.linear import FusedLinear, make_linear
from mini_engine.kernels.prefill import prefill_attention
from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config
from mini_engine.quantization import create_backend


def _tiny_config():
    # Non-TMA-aligned N and a short prompt exercise the decode-oriented routes.
    return Qwen3Config(71, 64, 128, 2, 4, 2, 16, max_position_embeddings=64)


@pytest.fixture
def dandan_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("dandan_ops inference GEMM requires SM80 or newer")
    if importlib.util.find_spec("dandan_ops") is None:
        pytest.skip("optional dandan_ops package is not installed")
    package = importlib.import_module("dandan_ops")
    # An installed but broken extension should fail rather than look like an
    # absent optional dependency or silently benchmark a reference fallback.
    importlib.import_module("dandan_ops._extension").load_extension(required=True)
    return package


def _forbid_hf_loading(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("invalid backend configuration reached HF loading")

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=unexpected),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=unexpected),
    ))


@pytest.fixture
def without_dandan_ops(monkeypatch):
    from mini_engine.kernels.dandan_backend import require_dandan_ops

    # Other tests may already have imported the real package and populated the
    # process-wide dependency cache. The missing-package test must not depend
    # on running first, and must not leave a cached module behind afterward.
    require_dandan_ops.cache_clear()
    monkeypatch.setitem(sys.modules, "dandan_ops", None)
    yield
    require_dandan_ops.cache_clear()


def test_torch_model_works_when_optional_dependency_is_missing(without_dandan_ops):
    torch.manual_seed(101)
    model = NativeQwen3(_tiny_config(), prefill_backend="math").eval()
    cache = model.create_cache(4, 4)
    logits = model(torch.tensor([3, 8, 5]), cache, [2, 0])
    assert logits.shape == (3, model.config.vocab_size)
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("backend_kwargs", [
    {"linear_backend": "dandan_cuda"},
    {"prefill_backend": "dandan"},
])
def test_explicit_dandan_request_fails_before_download_when_missing(monkeypatch, without_dandan_ops, backend_kwargs):
    _forbid_hf_loading(monkeypatch)
    with pytest.raises(ImportError, match="dandan"):
        NativeQwen3Runner.from_pretrained(
            "must-not-be-downloaded", device="cuda", dtype=torch.float16,
            local_files_only=False, **backend_kwargs,
        )


@pytest.mark.parametrize("kwargs", [
    {"device": "cpu", "dtype": torch.float16, "linear_backend": "dandan"},
    {"device": "cuda", "dtype": torch.bfloat16, "linear_backend": "dandan_cuda"},
    {"quant_backend": "unknown"},
    {"quant_backend": "dandan"},
    {"device": "cuda", "dtype": torch.float16, "quant_backend": "dandan",
     "quantization": QuantConfig("int8", activation_dtype="fp16", group_size=64)},
    {"device": "cuda", "dtype": torch.float16, "quant_backend": "dandan",
     "quantization": QuantConfig("int4", activation_dtype="bf16", group_size=64)},
    {"device": "cuda", "dtype": torch.bfloat16, "quant_backend": "dandan",
     "quantization": QuantConfig("int4", activation_dtype="fp16", group_size=64)},
    {"device": "cpu", "dtype": torch.float16, "quant_backend": "dandan",
     "quantization": QuantConfig("int4", activation_dtype="fp16", group_size=64)},
    {"device": "cuda", "dtype": torch.float16, "quant_backend": "dandan",
     "quantization": QuantConfig("int4", activation_dtype="fp16", group_size=32)},
    {"device": "cpu", "dtype": torch.float16, "prefill_backend": "dandan"},
    {"device": "cuda", "dtype": torch.bfloat16, "prefill_backend": "dandan"},
])
def test_unsupported_startup_options_fail_before_hf_loading(monkeypatch, kwargs):
    _forbid_hf_loading(monkeypatch)
    with pytest.raises((ValueError, NotImplementedError)):
        NativeQwen3Runner.from_pretrained("must-not-be-downloaded", **kwargs)


@pytest.mark.parametrize("missing_api,kwargs", [
    ("flash_attention", {"prefill_backend": "dandan"}),
    ("pack_int4", {"quant_backend": "dandan", "quantization": QuantConfig(
        "int4", activation_dtype="fp16", group_size=64)}),
])
def test_missing_operator_capability_fails_before_hf_loading(dandan_cuda, monkeypatch, missing_api, kwargs):
    _forbid_hf_loading(monkeypatch)
    monkeypatch.setattr(dandan_cuda, missing_api, None)
    with pytest.raises(ImportError, match=missing_api):
        NativeQwen3Runner.from_pretrained(
            "must-not-be-downloaded", device="cuda", dtype=torch.float16, **kwargs,
        )


def test_prefill_head_dimension_checked_before_tokenizer_and_model_weights(dandan_cuda, monkeypatch):
    _forbid_hf_loading(monkeypatch)
    config_calls = []

    def config_only(path, **kwargs):
        config_calls.append((path, kwargs))
        return SimpleNamespace(head_dim=64, hidden_size=256, num_attention_heads=4)

    monkeypatch.setattr(sys.modules["transformers"], "AutoConfig",
                        SimpleNamespace(from_pretrained=config_only), raising=False)
    with pytest.raises(ValueError, match="head_dim=128"):
        NativeQwen3Runner.from_pretrained(
            "config-only", device="cuda", dtype=torch.float16,
            prefill_backend="dandan", local_files_only=True,
        )
    assert config_calls == [("config-only", {"local_files_only": True})]


@pytest.mark.parametrize("backend", ["dandan", "dandan_cuda"])
@pytest.mark.parametrize("shape", [(1, 70), (2, 3, 70)])
def test_fp16_projection_and_fused_projection_match_cublas(dandan_cuda, backend, shape):
    torch.manual_seed(102)
    first = make_linear(70, 45, bias=True, backend=backend).cuda().half().eval()
    second = make_linear(70, 19, bias=True, backend=backend).cuda().half().eval()
    fused = FusedLinear.from_linears((first, second), backend=backend).eval()
    x = torch.randn(shape, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        expected = torch.cat((F.linear(x, first.weight, first.bias),
                              F.linear(x, second.weight, second.bias)), dim=-1)
        torch.testing.assert_close(first(x), expected[..., :45], atol=0.004, rtol=0.01)
        torch.testing.assert_close(fused(x), expected, atol=0.004, rtol=0.01)
    assert fused.backend == backend


def test_explicit_backend_does_not_hide_operator_failure(dandan_cuda, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected dandan GEMM failure")

    monkeypatch.setattr(dandan_cuda, "gemm", fail)
    linear = make_linear(64, 32, bias=False, backend="dandan_cuda").cuda().half().eval()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="injected dandan"):
        linear(torch.randn(2, 64, device="cuda", dtype=torch.float16))


@pytest.mark.parametrize("tokens", [17, 65, 129])
def test_dandan_prefill_native_gqa_matches_math_without_kv_expansion(dandan_cuda, monkeypatch, tokens):
    torch.manual_seed(106)
    q = torch.randn(2, 6, tokens, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 2, tokens, 128, device="cuda", dtype=torch.float16)
    v = torch.randn_like(k)
    expected = prefill_attention(q, k, v, "math")
    actual_flash = dandan_cuda.flash_attention
    calls = []

    def track_query_and_kv(query, key, value, *args, **kwargs):
        calls.append((query.shape, key.shape, value.shape, kwargs))
        return actual_flash(query, key, value, *args, **kwargs)

    monkeypatch.setattr(dandan_cuda, "flash_attention", track_query_and_kv)
    with torch.inference_mode():
        actual = prefill_attention(q, k, v, "dandan")
    torch.testing.assert_close(actual, expected, atol=0.004, rtol=0.02)
    assert len(calls) == 1
    assert calls[0][:3] == (q.shape, k.shape, v.shape)
    assert calls[0][3].get("backend") == "cuda"
    assert calls[0][3].get("causal") is True


@pytest.mark.parametrize("device,dtype,head_dim", [
    ("cpu", torch.float16, 128),
    ("cuda", torch.bfloat16, 128),
    ("cuda", torch.float16, 64),
])
def test_dandan_prefill_rejects_unsupported_device_dtype_and_head_dimension(dandan_cuda, device, dtype, head_dim):
    q = torch.randn(1, 4, 5, head_dim, device=device, dtype=dtype)
    k = torch.randn(1, 2, 5, head_dim, device=device, dtype=dtype)
    with torch.inference_mode(), pytest.raises(ValueError):
        prefill_attention(q, k, k, "dandan")


def test_dandan_prefill_does_not_hide_kernel_failure(dandan_cuda, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("injected dandan prefill failure")

    monkeypatch.setattr(dandan_cuda, "flash_attention", fail)
    q = torch.randn(1, 4, 5, 128, device="cuda", dtype=torch.float16)
    k = torch.randn(1, 2, 5, 128, device="cuda", dtype=torch.float16)
    with torch.inference_mode(), pytest.raises(RuntimeError, match="injected dandan prefill"):
        prefill_attention(q, k, k, "dandan")


@pytest.mark.parametrize("group_size", [64, 128])
@pytest.mark.parametrize("rows", [1, 4])
def test_int4_bridge_preserves_symmetric_state_and_fp16_oracle(dandan_cuda, group_size, rows):
    from mini_engine.quantization.dandan import DandanQuantLinear

    torch.manual_seed(103)
    config = QuantConfig("int4", activation_dtype="fp16", group_size=group_size)
    # K=70 covers partial 64- and 128-value groups and a partial packed word.
    source = nn.Linear(70, 19, bias=True, device="cuda", dtype=torch.float16).eval()
    with torch.no_grad():
        source.weight[0].zero_()
    reference_backend = create_backend(config)
    packed = reference_backend.pack(source.weight)
    restored = reference_backend.dequantize(packed)
    module = DandanQuantLinear(source, config).eval()
    torch.testing.assert_close(module.dequantize(), restored, atol=0, rtol=0)
    assert not list(module.parameters())
    assert module.storage_bytes < source.weight.numel() * source.weight.element_size()
    x = torch.randn(rows, 70, device="cuda", dtype=torch.float16)
    with torch.inference_mode():
        # The library applies bias after converting GEMM output back to FP16.
        expected = (x.float() @ restored.half().float().t()).half() + source.bias
        torch.testing.assert_close(module(x), expected, atol=0.004, rtol=0.01)
    route = module.op.execution_info(x)
    assert route["implementation"].startswith("w4a16_")
    assert route["fallback_reason"] is None

    clone = DandanQuantLinear(source, config).eval()
    clone.load_state_dict(copy.deepcopy(module.state_dict()))
    # Calling half() must not round FP32 scales or expand packed storage.
    scales = module.op.scales.clone()
    clone.half()
    assert clone.op.scales.dtype == torch.float32
    assert clone.op.packed_weight.dtype == torch.int32
    assert torch.equal(clone.op.scales, scales)
    with torch.inference_mode():
        torch.testing.assert_close(clone(x), module(x), atol=0, rtol=0)


@pytest.mark.parametrize("source_group", [64, 128])
def test_int4_checkpoint_rejects_different_source_group_metadata(dandan_cuda, source_group):
    from mini_engine.quantization.dandan import DandanQuantLinear

    # Both bridge outputs have identical physical shape and storage group=64.
    # Metadata must retain the source grouping to prevent a misleading reload.
    source = nn.Linear(70, 19, bias=False, device="cuda", dtype=torch.float16).eval()
    original = DandanQuantLinear(source, QuantConfig("int4", activation_dtype="fp16", group_size=source_group))
    other_group = 128 if source_group == 64 else 64
    incompatible = DandanQuantLinear(source, QuantConfig("int4", activation_dtype="fp16", group_size=other_group))
    assert original.op.packed_weight.shape == incompatible.op.packed_weight.shape
    assert original.op.scales.shape == incompatible.op.scales.shape
    with pytest.raises((ValueError, RuntimeError), match="bridge metadata/source group"):
        incompatible.load_state_dict(copy.deepcopy(original.state_dict()))


def _fp16_weight_oracle(model):
    """Replace only quantized projections using their *same* packed weights."""
    from mini_engine.quantization.dandan import DandanQuantLinear

    oracle = copy.deepcopy(model)
    selected = [(name, module) for name, module in oracle.named_modules()
                if isinstance(module, DandanQuantLinear)]
    assert selected
    for name, module in selected:
        weight = module.dequantize().half()
        linear = nn.Linear(module.in_features, module.out_features,
                           bias=module.bias is not None, device=weight.device, dtype=weight.dtype)
        with torch.no_grad():
            linear.weight.copy_(weight)
            if module.bias is not None:
                linear.bias.copy_(module.bias)
        parent, child = name.rsplit(".", 1)
        setattr(oracle.get_submodule(parent), child, linear)
    return oracle.eval()


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_tiny_model_prefill_cached_decode_and_kv_match_oracle(dandan_cuda, quantized, fused):
    torch.manual_seed(104)
    model = NativeQwen3(_tiny_config(), prefill_backend="math", linear_backend="dandan_cuda")
    model.cuda().half().eval()
    oracle = NativeQwen3(_tiny_config(), prefill_backend="math").cuda().half().eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    if fused:
        assert model.enable_projection_fusion() == 4
        oracle.enable_projection_fusion()
        assert all(layer.self_attn.qkv_proj.backend == "dandan_cuda" for layer in model.model.layers)
        assert all(layer.mlp.gate_up_proj.backend == "dandan_cuda" for layer in model.model.layers)
    if quantized:
        report = model.quantize(QuantConfig("int4", activation_dtype="fp16", group_size=128),
                                backend="dandan")
        assert report["accelerated_quant_gemm"] and not report["reference_only"]
        # Fusion reduces seven projection modules per layer to four.
        assert len(report["replaced_layers"]) == (8 if fused else 14)
        oracle = _fp16_weight_oracle(model)
    model_cache, oracle_cache = model.create_cache(8, 4), oracle.create_cache(8, 4)
    blocks = [6, 1, 4, 0]
    tokens = torch.tensor([3, 8, 11, 5, 9], device="cuda")
    with torch.inference_mode():
        actual = model(tokens, model_cache, blocks)
        expected = oracle(tokens, oracle_cache, blocks)
        torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.02)
        for position in range(5, 9):
            token = expected[-1].argmax().view(1)
            actual = model(token, model_cache, blocks, position)
            expected = oracle(token, oracle_cache, blocks, position)
            torch.testing.assert_close(actual, expected, atol=0.03, rtol=0.02)
            assert actual[-1].argmax() == expected[-1].argmax()
        for layer in range(model.config.num_hidden_layers):
            actual_k, actual_v = model_cache.gather(layer, blocks, 9)
            expected_k, expected_v = oracle_cache.gather(layer, blocks, 9)
            torch.testing.assert_close(actual_k, expected_k, atol=0.01, rtol=0.02)
            torch.testing.assert_close(actual_v, expected_v, atol=0.01, rtol=0.02)


@pytest.mark.parametrize("quantized", [False, True])
@pytest.mark.parametrize("fused", [False, True])
def test_d128_gqa_model_flash_prefill_and_decode_match_math(dandan_cuda, quantized, fused):
    torch.manual_seed(107)
    config = Qwen3Config(71, 128, 256, 2, 2, 1, 128, max_position_embeddings=64)
    model = NativeQwen3(config, prefill_backend="dandan", linear_backend="dandan_cuda").cuda().half().eval()
    oracle = NativeQwen3(config, prefill_backend="math").cuda().half().eval()
    oracle.load_state_dict(model.state_dict(), strict=True)
    if fused:
        model.enable_projection_fusion()
        oracle.enable_projection_fusion()
    if quantized:
        model.quantize(QuantConfig("int4", activation_dtype="fp16", group_size=128), backend="dandan")
        oracle = _fp16_weight_oracle(model)
        oracle.prefill_backend = "math"
    actual_cache, expected_cache = model.create_cache(8, 8), oracle.create_cache(8, 8)
    blocks = [6, 1, 4, 0]
    tokens = torch.tensor([3, 8, 11, 5, 9, 17, 21, 4, 6, 1, 10, 15, 7, 29, 2, 16, 19], device="cuda")
    with torch.inference_mode():
        actual = model(tokens, actual_cache, blocks)
        expected = oracle(tokens, expected_cache, blocks)
        torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.02)
        assert actual[-1].argmax() == expected[-1].argmax()
        for position in range(tokens.numel(), tokens.numel() + 3):
            token = expected[-1].argmax().view(1)
            actual = model(token, actual_cache, blocks, position)
            expected = oracle(token, expected_cache, blocks, position)
            torch.testing.assert_close(actual, expected, atol=0.06, rtol=0.02)
            assert actual[-1].argmax() == expected[-1].argmax()
        for layer in range(config.num_hidden_layers):
            actual_k, actual_v = actual_cache.gather(layer, blocks, tokens.numel() + 3)
            expected_k, expected_v = expected_cache.gather(layer, blocks, tokens.numel() + 3)
            torch.testing.assert_close(actual_k, expected_k, atol=0.012, rtol=0.02)
            torch.testing.assert_close(actual_v, expected_v, atol=0.012, rtol=0.02)


def test_native_dandan_prefill_rejects_model_with_unsupported_head_dimension(dandan_cuda):
    model = NativeQwen3(_tiny_config(), prefill_backend="dandan").cuda().half().eval()
    cache = model.create_cache(4, 4)
    with pytest.raises(ValueError):
        model(torch.tensor([3, 8, 5], device="cuda"), cache, [2, 0])


@pytest.mark.parametrize("padding,match", [
    ([9, 10, 11], "private page"),
    ([9, 10, 9, 12], "distinct"),
])
def test_graph_padding_rejects_insufficient_or_duplicate_private_pages(dandan_cuda, padding, match):
    from mini_engine.engine.cuda_graph import DecodeGraphBucket

    fake_cache = SimpleNamespace(device=torch.device("cuda"))
    with pytest.raises(ValueError, match=match):
        DecodeGraphBucket(SimpleNamespace(), fake_cache, 4, 3, padding_block_ids=padding)


@pytest.mark.skipif(os.environ.get("MINI_ENGINE_TEST_CUDA") != "1",
                    reason="set MINI_ENGINE_TEST_CUDA=1 to load engine CUDA kernels")
@pytest.mark.parametrize("quantized", [False, True])
def test_real_model_graph_replay_matches_eager_with_padding(dandan_cuda, quantized):
    from mini_engine.engine.cuda_graph import DecodeGraphBucket
    from mini_engine.kernels.cuda_extension import load_cuda_extension

    load_cuda_extension()  # Never compile or initialize kernels during capture.
    torch.manual_seed(105)
    model = NativeQwen3(_tiny_config(), prefill_backend="math", decode_backend="cuda",
                       data_backend="cuda", elementwise_backend="cuda", linear_backend="dandan_cuda")
    model.cuda().half().eval()
    model.enable_projection_fusion()
    if quantized:
        model.quantize(QuantConfig("int4", activation_dtype="fp16", group_size=64), backend="dandan")
    graph_cache, eager_cache = model.create_cache(16, 4), model.create_cache(16, 4)
    graph_cache.clear()
    eager_cache.clear()
    request_blocks = [[6, 1, 4], [8, 0, 5]]
    # Scratch blocks are disjoint from both requests, including future pages.
    bucket = DecodeGraphBucket(model, graph_cache, 4, 3, padding_block_ids=[9, 10, 11, 12])
    bucket.capture(warmup=2)
    assert bucket.block_tables[:, 0].tolist() == [9, 10, 11, 12]
    prompts = [torch.tensor([3, 8, 11, 5, 9], device="cuda"),
               torch.tensor([2, 7, 13], device="cuda")]
    with torch.inference_mode():
        for prompt, blocks in zip(prompts, request_blocks):
            model(prompt, graph_cache, blocks)
            model(prompt, eager_cache, blocks)
        positions = torch.tensor([5, 3], device="cuda", dtype=torch.long)
        tokens = torch.tensor([17, 19], device="cuda", dtype=torch.long)
        tables = torch.tensor(request_blocks, device="cuda", dtype=torch.long)
        for active in (2, 2, 1):
            expected = model.forward_decode_batch(tokens[:active], eager_cache, tables[:active], positions[:active])
            actual = bucket.replay(tokens[:active], positions[:active], tables[:active], active_count=active)
            torch.testing.assert_close(actual, expected, atol=0.01, rtol=0.01)
            assert torch.equal(actual.argmax(-1), expected.argmax(-1))
            assert bucket.block_tables[active:, 0].tolist() == [9, 10, 11, 12][active:]
            assert bucket.block_tables[:, 0].unique().numel() == 4
            # Compare live cache contents after every replay, including when
            # a padding slot replaces the second request.
            for row in range(2):
                count = int(positions[row]) + (row < active)
                for layer in range(model.config.num_hidden_layers):
                    actual_k, actual_v = graph_cache.gather(layer, request_blocks[row], count)
                    expected_k, expected_v = eager_cache.gather(layer, request_blocks[row], count)
                    torch.testing.assert_close(actual_k, expected_k, atol=0.005, rtol=0.01)
                    torch.testing.assert_close(actual_v, expected_v, atol=0.005, rtol=0.01)
            tokens[:active] = expected.argmax(-1)
            positions[:active] += 1
