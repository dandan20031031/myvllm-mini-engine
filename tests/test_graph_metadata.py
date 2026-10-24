"""Reusable CPU staging contracts and opt-in graph metadata replay checks."""

import os
from types import SimpleNamespace

import pytest
import torch

from mini_engine.engine.cuda_graph import DecodeGraphBucket


def test_python_metadata_validation_accepts_variable_width_rows():
    assert DecodeGraphBucket.validate_metadata(
        [3, 5], [0, 7], [[9], [2, 4]], batch_size=4, max_blocks=3,
    ) == 2


@pytest.mark.parametrize("tokens,positions,tables", [
    ([], [], []),
    ([1, 2, 3], [0, 0, 0], [[0], [1], [2]]),
    ([1, 2], [0], [[0], [1]]),
    ([1], [0], []),
    ([1], [-1], [[0]]),
    ([-1], [0], [[0]]),
    ([True], [0], [[0]]),
    ([1], [False], [[0]]),
    ([1], [0], [[True]]),
    ([1], [0], [[-2]]),
    ([1], [0], [[]]),
    ([1], [0], [[0, 1, 2, 3]]),
    ([1.5], [0], [[0]]),
])
def test_python_metadata_rejects_invalid_shape_or_values(tokens, positions, tables):
    with pytest.raises((ValueError, TypeError)):
        DecodeGraphBucket.validate_metadata(tokens, positions, tables, batch_size=2, max_blocks=3)


CUDA_METADATA = pytest.mark.skipif(
    os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set MINI_ENGINE_TEST_CUDA=1 with CUDA available",
)


def make_bucket(*, sampling=False):
    class FakeModel:
        config = SimpleNamespace(vocab_size=17, max_position_embeddings=64)

        def forward_decode_batch(self, tokens, cache, tables, positions):
            center = ((tokens + positions) % self.config.vocab_size).float().unsqueeze(1)
            vocab = torch.arange(self.config.vocab_size, device=tokens.device).float()
            return -(vocab.unsqueeze(0) - center).square() + tables[:, :1].float() * 0.01

    cache = SimpleNamespace(device=torch.device("cuda"), shape=SimpleNamespace(num_blocks=32, block_size=4))
    bucket = DecodeGraphBucket(FakeModel(), cache, 4, 3, padding_block_ids=[24, 25, 26, 27],
                               capture_sampling=sampling, sampling_backend="torch")
    bucket.capture(warmup=1)
    return bucket


@CUDA_METADATA
def test_metadata_staging_reuses_addresses_and_clears_shorter_rows_and_slots():
    bucket = make_bucket()
    names = ("host_input_ids", "host_positions", "host_block_tables", "input_ids", "positions", "block_tables")
    pointers = {name: getattr(bucket, name).data_ptr() for name in names}
    assert bucket.host_input_ids.is_pinned()
    assert bucket.host_positions.is_pinned()
    assert bucket.host_block_tables.is_pinned()
    cases = [
        ([3, 5, 7, 9], [0, 4, 8, 1], [[1], [2, 3], [4, 5, 6], [7]]),
        ([11], [0], [[8]]),
        ([2, 4, 6], [3, 5, 0], [[9], [10, 11], [12]]),
    ]
    for tokens, positions, tables in cases:
        actual = bucket.replay_metadata(tokens, positions, tables).clone()
        ids = torch.tensor(tokens, device="cuda")
        offsets = torch.tensor(positions, device="cuda")
        table = torch.full((len(tokens), 3), -1, dtype=torch.long, device="cuda")
        for row, pages in enumerate(tables):
            table[row, :len(pages)] = torch.tensor(pages, device="cuda")
        expected = bucket.model.forward_decode_batch(ids, bucket.cache, table, offsets)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.cuda.synchronize()
        for row, pages in enumerate(tables):
            assert bucket.block_tables[row].tolist() == pages + [-1] * (3 - len(pages))
        for row in range(len(tokens), 4):
            assert bucket.block_tables[row].tolist() == [24 + row, -1, -1]
        assert pointers == {name: getattr(bucket, name).data_ptr() for name in names}


@CUDA_METADATA
def test_metadata_replay_can_return_captured_sampling_tokens():
    bucket = make_bucket(sampling=True)
    output = bucket.replay_metadata([3, 11], [0, 5], [[1], [2, 3]], sample=True)
    assert output.dtype == torch.long and output.shape == (2,)
    assert output.cpu().tolist() == [3, 16]
    assert bucket.sampled_token_ids is not None
    next_output = bucket.replay_metadata([4], [8], [[4, 5, 6]], sample=True)
    assert next_output.cpu().tolist() == [12]


@CUDA_METADATA
def test_metadata_rejects_sampling_without_capture_and_bad_physical_capacity():
    bucket = make_bucket()
    with pytest.raises((ValueError, RuntimeError), match="sampl"):
        bucket.replay_metadata([3], [0], [[1]], sample=True)
    for tokens, positions, tables in (([3], [0], [[32]]), ([3], [8], [[1]]), ([17], [0], [[1]])):
        with pytest.raises((ValueError, IndexError)):
            bucket.replay_metadata(tokens, positions, tables)
