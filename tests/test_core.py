import torch

from mini_engine.cache.block_manager import BlockManager
from mini_engine.engine.request import GenerationRequest, RequestState
from mini_engine.engine.scheduler import Scheduler
from mini_engine.kernels.fallback import rms_norm
from mini_engine.sampling import sample_next_token


def test_block_manager_grows_and_releases():
    manager = BlockManager(num_blocks=4, block_size=4)
    assert manager.allocate("a", 5) == [0, 1]
    assert manager.grow("a", 9) == [0, 1, 2]
    manager.release("a")
    assert manager.free_blocks == 4


def test_scheduler_admits_prefill_then_decode():
    manager = BlockManager(num_blocks=8, block_size=4)
    scheduler = Scheduler(manager, max_num_batched_tokens=32)
    request = GenerationRequest("a", [1, 2, 3, 4, 5], max_new_tokens=2)
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert [r.request_id for r in first.prefill] == ["a"]
    assert request.state is RequestState.RUNNING
    second = scheduler.schedule()
    assert [r.request_id for r in second.decode] == ["a"]
    scheduler.on_token(request, 9)
    scheduler.on_token(request, 10)
    assert request.is_finished
    assert manager.free_blocks == 8


def test_fallback_ops_and_greedy_sampling():
    x = torch.ones(2, 4)
    y = rms_norm(x, torch.ones(4))
    assert torch.allclose(y, torch.ones_like(y), atol=1e-5)
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    assert sample_next_token(logits).item() == 1
