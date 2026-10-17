"""Opt-in Windows CUDA coverage: MINI_ENGINE_TEST_CUDA=1."""

import os

import pytest
import torch

from mini_engine.kernels.elementwise import residual_add, residual_rms_norm, rms_norm, silu_and_mul

pytestmark = pytest.mark.skipif(
    os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set MINI_ENGINE_TEST_CUDA=1 with MSVC/nvcc/Ninja available",
)

DTYPES = [torch.float16, torch.bfloat16]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(1, 1), (3, 17), (2, 5, 128), (3, 1024), (5, 3072), (2, 12345)])
def test_cuda_norm_and_fused_residual(dtype, shape):
    torch.manual_seed(31)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(shape[-1], device="cuda", dtype=dtype)
    snapshots = x.clone(), residual.clone()
    expected = rms_norm(x, weight, backend="torch")
    actual = rms_norm(x, weight, backend="cuda")
    expected_norm, expected_sum = residual_rms_norm(x, residual, weight, backend="torch")
    actual_norm, actual_sum = residual_rms_norm(x, residual, weight, backend="cuda")
    atol = 0.004 if dtype == torch.bfloat16 else 0.001
    torch.testing.assert_close(actual, expected, atol=atol, rtol=0.008)
    torch.testing.assert_close(actual_norm, expected_norm, atol=atol, rtol=0.008)
    assert torch.equal(actual_sum, expected_sum)
    assert torch.equal(residual_add(x, residual, backend="cuda"), expected_sum)
    assert torch.equal(x, snapshots[0]) and torch.equal(residual, snapshots[1])


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_cuda_rounding_cancellation_zeros_and_small_input(dtype, eps):
    weight = torch.tensor([0.125, -2.0, 3.5, 1.0], device="cuda", dtype=dtype)
    x = torch.tensor([[0.0, 0.0, 0.0, 0.0], [1e-5, -1e-5, 1e-6, 0.0],
                      [0.5078125, 3.140625, -2.71875, 1.25]], device="cuda", dtype=dtype)
    residual = -x
    expected = rms_norm(x, weight, eps, backend="torch")
    actual = rms_norm(x, weight, eps, backend="cuda")
    torch.testing.assert_close(actual, expected, atol=0.001, rtol=0.008)
    normalized, summed = residual_rms_norm(x, residual, weight, eps, backend="cuda")
    assert not normalized.any() and not summed.any()


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("shape", [(), (0,), (4, 129), (2, 3072)])
def test_cuda_silu_mul(dtype, shape):
    torch.manual_seed(32)
    gate = torch.randn(shape, device="cuda", dtype=dtype)
    up = torch.randn_like(gate)
    actual = silu_and_mul(gate, up, backend="cuda")
    expected = silu_and_mul(gate, up, backend="torch")
    torch.testing.assert_close(actual, expected, atol=0.004 if dtype == torch.bfloat16 else 0.001, rtol=0.008)
    assert torch.equal(residual_add(gate, up, backend="cuda"), gate + up)


@pytest.mark.parametrize("dtype", DTYPES)
def test_empty_batch_extreme_activation_and_current_stream(dtype):
    gate = torch.tensor([-100.0, -20, -1, 0, 1, 20, 100], device="cuda", dtype=dtype)
    up = torch.ones_like(gate)
    # Establish correct input readiness before entering the side stream.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    weight = torch.ones(7, device="cuda", dtype=dtype)
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = silu_and_mul(gate, up, backend="cuda")
        norm = rms_norm(gate, weight, backend="cuda")
        fused, summed = residual_rms_norm(gate, up, weight, backend="cuda")
        added = residual_add(gate, up, backend="cuda")
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, silu_and_mul(gate, up), atol=0.004, rtol=0.008)
    torch.testing.assert_close(norm, rms_norm(gate, weight), atol=0.004, rtol=0.008)
    torch.testing.assert_close(fused, rms_norm(gate + up, weight), atol=0.004, rtol=0.008)
    assert torch.equal(summed, added)
    empty = torch.empty(0, 7, device="cuda", dtype=dtype)
    assert rms_norm(empty, weight, backend="cuda").shape == empty.shape
    a, b = residual_rms_norm(empty, empty, weight, backend="cuda")
    assert a.numel() == b.numel() == 0


@pytest.mark.parametrize("case", ["cpu", "dtype", "strided", "scalar", "empty_hidden", "weight_shape",
                                  "weight_device", "pair_dtype", "pair_shape", "eps_zero", "eps_nan",
                                  "eps_inf", "eps_tiny", "autograd"])
def test_binding_rejects_invalid_inputs(case):
    from mini_engine.kernels.cuda_extension import load_cuda_extension
    ext = load_cuda_extension()
    x = torch.randn(2, 8, device="cuda", dtype=torch.float16)
    w = torch.ones(8, device="cuda", dtype=torch.float16)
    if case == "cpu":
        fn = lambda: ext.rms_norm(x.cpu(), w.cpu(), 1e-6)
    elif case == "dtype":
        fn = lambda: ext.rms_norm(x.float(), w.float(), 1e-6)
    elif case == "strided":
        fn = lambda: ext.rms_norm(x[:, ::2], w[:4], 1e-6)
    elif case == "scalar":
        fn = lambda: ext.rms_norm(x[0, 0], w[:1], 1e-6)
    elif case == "empty_hidden":
        fn = lambda: ext.rms_norm(x[:, :0], w[:0], 1e-6)
    elif case == "weight_shape":
        fn = lambda: ext.rms_norm(x, w[:7], 1e-6)
    elif case == "weight_device":
        fn = lambda: ext.rms_norm(x, w.cpu(), 1e-6)
    elif case == "pair_dtype":
        fn = lambda: ext.residual_rms_norm(x, x.bfloat16(), w, 1e-6)
    elif case == "pair_shape":
        fn = lambda: ext.silu_and_mul(x, x[:1])
    elif case == "autograd":
        fn = lambda: ext.rms_norm(x.requires_grad_(), w, 1e-6)
    else:
        eps = {"eps_zero": 0.0, "eps_nan": float("nan"), "eps_inf": float("inf"), "eps_tiny": 1e-100}[case]
        fn = lambda: ext.rms_norm(x, w, eps)
    with pytest.raises(RuntimeError):
        fn()


@pytest.mark.parametrize("dtype", DTYPES)
def test_cuda_graph_replay(dtype):
    from mini_engine.kernels.cuda_extension import load_cuda_extension
    load_cuda_extension()
    x = torch.randn(3, 128, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(128, device="cuda", dtype=dtype)
    def run():
        norm, summed = residual_rms_norm(x, residual, weight, backend="cuda")
        activated = silu_and_mul(norm, summed, backend="cuda")
        return rms_norm(residual_add(activated, summed, backend="cuda"), weight, backend="cuda")
    for _ in range(3):
        run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = run()
    x.mul_(0.5)
    graph.replay()
    norm, summed = residual_rms_norm(x, residual, weight)
    expected = rms_norm(residual_add(silu_and_mul(norm, summed), summed), weight)
    torch.testing.assert_close(output, expected, atol=0.01 if dtype == torch.bfloat16 else 0.002, rtol=0.01)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("quantization", [None, "int4", "fp8_e4m3"])
def test_native_hf_prefill_decode_and_elementwise_call_coverage(dtype, quantization, monkeypatch):
    pytest.importorskip("transformers")
    from transformers import Qwen3Config as HFConfig, Qwen3ForCausalLM
    from mini_engine.config import QuantConfig
    from mini_engine.model import NativeQwen3
    from mini_engine.model import native_qwen3 as module
    torch.manual_seed(17)
    cfg = HFConfig(vocab_size=71, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                   num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                   max_position_embeddings=128, tie_word_embeddings=True, attention_dropout=0)
    cfg._attn_implementation = "eager"
    reference = Qwen3ForCausalLM(cfg).to(device="cuda", dtype=dtype).eval()
    # Non-unit norm weights exercise the intermediate cast before weight mul.
    with torch.no_grad():
        for name, parameter in reference.named_parameters():
            if "norm" in name:
                parameter.uniform_(0.5, 1.5)
    model = NativeQwen3.from_hf_model(reference, "math", "cuda", elementwise_backend="cuda")
    if quantization:
        act_dtype = "fp16" if dtype == torch.float16 else "bf16"
        model.quantize(QuantConfig(quantization, activation_dtype=act_dtype, group_size=8))
    # Independent native torch copy for comparing the exact same packed weights.
    baseline = NativeQwen3.from_hf_model(reference, "math", "torch", elementwise_backend="torch")
    if quantization:
        baseline.quantize(QuantConfig(quantization, activation_dtype=act_dtype, group_size=8))
    calls = {name: 0 for name in ["rms_norm", "residual_rms_norm", "residual_add", "silu_and_mul"]}
    def spy(name):
        original = getattr(module, name)
        def wrapped(*args, **kwargs):
            if kwargs.get("backend") == "cuda":
                calls[name] += 1
            return original(*args, **kwargs)
        return wrapped
    for name in calls:
        monkeypatch.setattr(module, name, spy(name))
    cache, base_cache = model.create_cache(8, 4), baseline.create_cache(8, 4)
    blocks = [6, 1, 4, 0]
    ids = torch.tensor([3, 8, 11, 5, 9], device="cuda")
    with torch.inference_mode():
        a, b = model(ids, cache, blocks), baseline(ids, base_cache, blocks)
        tolerance = 0.004 if dtype == torch.float16 else 0.025
        torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
        if quantization is None:
            torch.testing.assert_close(a, reference(ids[None], use_cache=False).logits[0],
                                       atol=tolerance, rtol=tolerance)
        for position in range(5, 11):
            token = b[-1].argmax().view(1)
            ids = torch.cat([ids, token])
            a = model(token, cache, blocks, position)
            b = baseline(token, base_cache, blocks, position)
            torch.testing.assert_close(a, b, atol=tolerance, rtol=tolerance)
            assert a[-1].argmax() == b[-1].argmax()
            if quantization is None:
                torch.testing.assert_close(a[-1], reference(ids[None], use_cache=False).logits[0, -1],
                                           atol=tolerance, rtol=tolerance)
    # 2 layers * (input + Q/K norms), plus final norm; fused post norm once/layer.
    assert calls == {"rms_norm": 49, "residual_rms_norm": 14, "residual_add": 14, "silu_and_mul": 14}


@pytest.mark.parametrize("dtype", DTYPES)
def test_contiguous_but_unaligned_views_and_inference_runner(dtype):
    from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config
    from test_native_qwen3 import TinyTokenizer
    # Contiguous does not imply 8-byte aligned when storage_offset is nonzero.
    x = torch.randn(257, device="cuda", dtype=dtype)[1:]
    other = torch.randn(257, device="cuda", dtype=dtype)[1:]
    assert x.is_contiguous() and x.storage_offset() == 1
    torch.testing.assert_close(silu_and_mul(x, other, backend="cuda"), silu_and_mul(x, other),
                               atol=0.004, rtol=0.008)
    assert torch.equal(residual_add(x, other, backend="cuda"), x + other)
    torch.manual_seed(2)
    model = NativeQwen3(Qwen3Config(37, 32, 64, 1, 4, 2, 16), "math", "cuda", "cuda").to(device="cuda", dtype=dtype)
    runner = NativeQwen3Runner(model, TinyTokenizer(), num_blocks=8, block_size=4)
    result = runner.generate_incremental("hi", 4)
    assert len(result.token_ids) == 4 and runner.block_manager.free_blocks == 8
