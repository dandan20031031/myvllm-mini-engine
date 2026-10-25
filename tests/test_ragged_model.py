"""Packed prefill matches independent per-request execution and KV ownership."""

import copy
import os

import pytest
import torch

from mini_engine.engine.batch import PrefillBatch
from mini_engine.model import NativeQwen3, Qwen3Config


def make_model(*, device="cpu", dtype=torch.float32, cuda=False):
    torch.manual_seed(812)
    config = Qwen3Config(43, 32, 64, 2, 4, 2, 16, max_position_embeddings=32)
    return NativeQwen3(config, "cuda" if cuda else "math", "cuda" if cuda else "torch",
        "cuda" if cuda else "torch", "cuda" if cuda else "torch",
        "cuda" if cuda else "torch").to(device=device, dtype=dtype).eval()


@torch.inference_mode()
def legacy_forward(model, tokens, cache, blocks, position=0):
    """Use the original single-request layer entry, not packed model forward."""
    weight, config = model.model.embed_tokens.weight, model.config
    ids = torch.tensor(tokens, device=weight.device, dtype=torch.int64)
    x = model.model.embed_tokens(ids)
    frequencies = 1.0 / (config.rope_theta ** (
        torch.arange(0, config.head_dim, 2, device=weight.device).float() / config.head_dim))
    positions = torch.arange(position, position + len(tokens), device=weight.device).float()
    angles = positions[:, None] * frequencies[None, :]
    cos, sin = angles.cos().to(x.dtype).unsqueeze(1), angles.sin().to(x.dtype).unsqueeze(1)
    for layer_id, layer in enumerate(model.model.layers):
        x = layer(x, cos, sin, cache, blocks, position, layer_id, model.prefill_backend, model.decode_backend)
    return model.lm_head(model.model.norm(x))


def metadata(sequences, tables, starts, *, device="cpu", num_blocks=16):
    return PrefillBatch.from_sequences(sequences, tables, starts, block_size=4,
        num_blocks=num_blocks, device=device)


def test_prefill_metadata_positions_slots_and_readonly_prefix_sharing():
    batch = metadata([[5, 6, 7], [8, 9]], [[3, 1, -1], [3, 2]], [4, 4])
    assert batch.query_lengths == (3, 2)
    assert batch.query_offsets == (0, 3, 5)
    assert batch.starts == (4, 4)
    assert batch.block_ids == ((3, 1), (3, 2))
    assert batch.block_size == 4 and batch.batch_size == 2 and batch.num_tokens == 5
    assert batch.input_ids.tolist() == [5, 6, 7, 8, 9]
    assert batch.positions.tolist() == [4, 5, 6, 4, 5]
    assert batch.slot_mapping.tolist() == [4, 5, 6, 8, 9]
    assert batch.cu_seqlens_q.tolist() == [0, 3, 5]
    assert batch.context_lens.tolist() == [7, 6]


@pytest.mark.parametrize("sequences,tables,starts,error", [
    ([[]], [[0]], [0], "nonempty"),
    ([[1]], [[0]], [-1], "nonnegative"),
    ([[1]], [[0]], [True], "integers"),
    ([[1]], [[0]], [], "equal batch"),
    ([[-1]], [[0]], [0], "nonnegative"),
    ([[1]], [[16]], [0], "out of range"),
    ([[1]], [[0, 0]], [0], "duplicate physical"),
    ([[1]], [[0, -1, 2]], [0], "trailing"),
    ([[1, 2]], [[0]], [3], "capacity"),
    ([[1], [2]], [[0], [0]], [0, 0], "duplicate writable"),
    ([[1], [2]], [[0], [0]], [0, 1], "shared by requests"),
    ([torch.tensor([1.0])], [[0]], [0], "int64"),
    ([[1 << 64]], [[0]], [0], "int64"),
])
def test_prefill_metadata_rejects_invalid_layout_before_execution(sequences, tables, starts, error):
    with pytest.raises(ValueError, match=error):
        metadata(sequences, tables, starts)


def test_ragged_mixed_prefix_positions_cross_pages_and_match_legacy_gqa():
    model = make_model()
    actual_cache, oracle_cache = model.create_cache(16, 4), model.create_cache(16, 4)
    tokens = [[1, 2, 3, 4, 5, 6, 7, 8, 9], [8, 3, 6, 2, 7],
              [3, 5, 7, 9, 2, 4, 6, 8], [2, 3, 4, 5, 6, 7, 8]]
    starts = [5, 0, 4, 6]
    tables = [[7, 1, 12], [9, 4], [3, 11], [15, 0]]
    snapshots = []
    for sequence, start, table in zip(tokens, starts, tables):
        if start:
            legacy_forward(model, sequence[:start], actual_cache, table)
            legacy_forward(model, sequence[:start], oracle_cache, table)
        snapshots.append([tuple(value.clone() for value in actual_cache.gather(layer, table, start))
                          for layer in range(model.config.num_hidden_layers)])
    batch = metadata([sequence[start:] for sequence, start in zip(tokens, starts)], tables, starts)
    expected = torch.cat([legacy_forward(model, sequence[start:], oracle_cache, table, start)
                          for sequence, start, table in zip(tokens, starts, tables)])
    actual = model.forward_prefill_batch(batch, actual_cache)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for request, (sequence, start, table) in enumerate(zip(tokens, starts, tables)):
        for layer in range(model.config.num_hidden_layers):
            actual_kv = actual_cache.gather(layer, table, len(sequence))
            expected_kv = oracle_cache.gather(layer, table, len(sequence))
            for actual_value, expected_value in zip(actual_kv, expected_kv):
                torch.testing.assert_close(actual_value, expected_value, atol=2e-6, rtol=2e-5)
            for current, original in zip(actual_cache.gather(layer, table, start), snapshots[request][layer]):
                torch.testing.assert_close(current, original, atol=0, rtol=0)


def test_shared_prefix_pages_stay_unchanged_and_requests_do_not_attend_to_each_other():
    model = make_model()
    cache = model.create_cache(16, 4)
    legacy_forward(model, [1, 2, 3, 4], cache, [0])
    prefix_key, prefix_value = cache.key[:, 0].clone(), cache.value[:, 0].clone()
    first = metadata([[5, 6, 7, 8, 9], [10, 11, 12]], [[0, 7, 4], [0, 2]], [4, 4])
    before = model.forward_prefill_batch(first, cache)
    changed = metadata([[5, 6, 7, 8, 9], [22, 23, 24]], [[0, 7, 4], [0, 2]], [4, 4])
    after = model.forward_prefill_batch(changed, cache)
    torch.testing.assert_close(before[:5], after[:5], atol=0, rtol=0)
    assert not torch.allclose(before[5:], after[5:])
    torch.testing.assert_close(cache.key[:, 0], prefix_key, atol=0, rtol=0)
    torch.testing.assert_close(cache.value[:, 0], prefix_value, atol=0, rtol=0)


def test_packed_rows_can_exceed_individual_context_limit_and_keep_module_hooks():
    model = make_model()
    cache = model.create_cache(16, 4)
    batch = metadata([list(range(1, 21)), list(range(20, 0, -1))],
                     [[0, 3, 5, 8, 10], [1, 4, 6, 9, 11]], [0, 0])
    calls, projection_shapes = [], []
    handles = [model.register_forward_hook(lambda *args: calls.append(1)),
               model.model.layers[0].self_attn.q_proj.register_forward_pre_hook(
                   lambda module, args: projection_shapes.append(tuple(args[0].shape)))]
    try:
        logits = model.forward_prefill_batch(batch, cache)
    finally:
        for handle in handles:
            handle.remove()
    assert logits.shape == (40, model.config.vocab_size)
    assert calls == [1] and projection_shapes == [(40, model.config.hidden_size)]


def test_selected_logits_preserve_order_duplicates_and_skip_empty_lm_head():
    model = make_model()
    cache = model.create_cache(16, 4)
    batch = metadata([[1, 2, 3], [4, 5]], [[2], [6]], [0, 0])
    full = model.forward_prefill_batch(batch, cache)
    selected = model.forward_prefill_batch(batch, cache, logits_rows=[4, 0, 2, 0])
    torch.testing.assert_close(selected, full[[4, 0, 2, 0]], atol=2e-6, rtol=2e-5)
    calls = []
    handle = model.lm_head.register_forward_pre_hook(lambda *args: calls.append(1))
    try:
        empty = model.forward_prefill_batch(batch, cache, logits_rows=[])
    finally:
        handle.remove()
    assert empty.shape == (0, model.config.vocab_size) and calls == []
    for actual, expected in zip(cache.gather(0, [2], 3),
                               (cache.key[0, 2, :3], cache.value[0, 2, :3])):
        torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("rows", [[-1], [5], [True], [1.0], torch.tensor([[0]]), torch.tensor([0.0])])
def test_bad_selected_rows_fail_before_writing_kv(rows):
    model = make_model()
    cache = model.create_cache(16, 4)
    cache.clear()
    batch = metadata([[1, 2, 3], [4, 5]], [[2], [6]], [0, 0])
    with pytest.raises(ValueError):
        model.forward_prefill_batch(batch, cache, logits_rows=rows)
    assert torch.count_nonzero(cache.key) == torch.count_nonzero(cache.value) == 0


def test_metadata_mutation_is_rejected_even_when_factory_runs_under_inference_mode():
    model = make_model()
    cache = model.create_cache(16, 4)
    with torch.inference_mode():
        batch = metadata([[1, 2]], [[3]], [0])
    batch.positions.add_(1)
    with pytest.raises(ValueError, match="modified"):
        model.forward_prefill_batch(batch, cache)


def test_b1_selected_single_query_at_nonzero_position_remains_prefill_compatible():
    model = make_model()
    cache, reference_cache = model.create_cache(16, 4), model.create_cache(16, 4)
    model(torch.tensor([1, 2, 3, 4]), cache, [7, 2])
    legacy_forward(model, [1, 2, 3, 4], reference_cache, [7, 2])
    expected = legacy_forward(model, [5], reference_cache, [7, 2], 4)
    actual = model(torch.tensor([5]), cache, [7, 2], position=4, logits_rows=[0])
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize("backend", ["flash", "dandan"])
def test_explicit_dense_backend_is_preserved_per_request(monkeypatch, backend):
    import mini_engine.model.native_qwen3 as implementation
    from mini_engine.kernels.prefill import prefill_attention

    model = make_model()
    model.prefill_backend = backend
    cache = model.create_cache(16, 4)
    batch = metadata([[1, 2, 3], [4, 5]], [[2], [6]], [0, 0])
    seen = []

    def record(q, k, v, selected, *, query_start=0):
        seen.append(selected)
        return prefill_attention(q, k, v, "math", query_start=query_start)

    monkeypatch.setattr(implementation, "prefill_attention", record)
    model.forward_prefill_batch(batch, cache)
    assert seen == [backend] * (batch.batch_size * model.config.num_hidden_layers)


CUDA_RAGGED = pytest.mark.skipif(
    os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set MINI_ENGINE_TEST_CUDA=1 with CUDA available",
)


@CUDA_RAGGED
@pytest.mark.parametrize("dtype,atol", [(torch.float16, 0.006), (torch.bfloat16, 0.04)])
def test_cuda_ragged_paged_prefix_selected_logits_match_torch(dtype, atol):
    reference = make_model(device="cuda", dtype=dtype)
    model = copy.deepcopy(reference)
    model.prefill_backend = model.decode_backend = model.data_backend = model.elementwise_backend = "cuda"
    for layer in model.model.layers:
        layer.elementwise_backend = layer.self_attn.data_backend = "cuda"
        layer.self_attn.q_norm.backend = layer.self_attn.k_norm.backend = "cuda"
        layer.input_layernorm.backend = layer.post_attention_layernorm.backend = "cuda"
        layer.mlp.elementwise_backend = "cuda"
    model.model.norm.backend = "cuda"
    cache, reference_cache = model.create_cache(16, 4), reference.create_cache(16, 4)
    tables, starts = [[7, 2, 11], [7, 5]], [4, 4]
    legacy_forward(reference, [1, 2, 3, 4], cache, [7])
    legacy_forward(reference, [1, 2, 3, 4], reference_cache, [7])
    prefix = (cache.key[:, 7].clone(), cache.value[:, 7].clone())
    sequences = [[5, 6, 7, 8, 9], [10, 11, 12]]
    batch = metadata(sequences, tables, starts, device="cuda")
    expected = torch.cat([legacy_forward(reference, tokens, reference_cache, blocks, start)
                          for tokens, blocks, start in zip(sequences, tables, starts)])
    actual = model.forward_prefill_batch(batch, cache, logits_rows=[4, 7])
    torch.testing.assert_close(actual, expected[[4, 7]], atol=atol, rtol=atol)
    torch.testing.assert_close(cache.key[:, 7], prefix[0], atol=0, rtol=0)
    torch.testing.assert_close(cache.value[:, 7], prefix[1], atol=0, rtol=0)
