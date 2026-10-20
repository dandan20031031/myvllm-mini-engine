import pytest
import torch

from mini_engine.engine.cuda_graph import DecodeGraphBucket, DecodeGraphManager, GraphShape


def test_graph_shape_bucket_selection_cpu():
    class FakeModel:
        pass

    class FakeCache:
        device = torch.device("cpu")

    manager = DecodeGraphManager(FakeModel(), FakeCache(), batch_sizes=(1, 2, 4), block_buckets=(8, 16))
    assert manager.shape_for(1, 3) == GraphShape(1, 8)
    assert manager.shape_for(3, 9) == GraphShape(4, 16)
    with pytest.raises(ValueError):
        manager.shape_for(5, 9)


def test_graph_bucket_requires_cuda_and_padding():
    class FakeModel:
        pass

    class FakeCache:
        device = torch.device("cpu")

    with pytest.raises(ValueError, match="CUDA"):
        DecodeGraphBucket(FakeModel(), FakeCache(), 2, 8)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_graph_bucket_replay_fixed_buffers():
    class FakeModel:
        def forward_decode_batch(self, input_ids, cache, block_tables, positions):
            return input_ids.float().unsqueeze(1) + positions.float().unsqueeze(1)

    class FakeCache:
        device = torch.device("cuda")

    manager = DecodeGraphManager(FakeModel(), FakeCache(), batch_sizes=(1, 2, 4), block_buckets=(8,))
    bucket = manager.capture(2, 4, padding_block_ids=list(range(8)), warmup=2)
    ids = torch.tensor([3, 7], device="cuda", dtype=torch.long)
    positions = torch.tensor([4, 9], device="cuda", dtype=torch.long)
    tables = torch.zeros(2, 8, device="cuda", dtype=torch.long)
    output = bucket.replay(ids, positions, tables, active_count=2)
    torch.testing.assert_close(output, (ids + positions).float().unsqueeze(1))
    single = bucket.replay(ids[:1], positions[:1], tables[:1], active_count=1)
    torch.testing.assert_close(single, (ids[:1] + positions[:1]).float().unsqueeze(1))
    assert bucket.input_ids.data_ptr() != 0 and bucket.block_tables.data_ptr() != 0
