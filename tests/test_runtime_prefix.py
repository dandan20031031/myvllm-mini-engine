import torch

from mini_engine.cache import BlockManager, PrefixCache
from mini_engine.engine.request import GenerationRequest, RequestState
from mini_engine.engine.runtime import NativeBatchRuntime
from mini_engine.engine.scheduler import Scheduler
from mini_engine.model import NativeQwen3, Qwen3Config


def test_prefix_cache_shares_and_lru_releases_blocks():
    manager = BlockManager(4, 4)
    cache = PrefixCache(manager, max_entries=1)
    first = manager.allocate("first", 4)
    entry = cache.insert([1, 2, 3, 4], first)
    assert entry is not None and manager.refcount(first[0]) == 2
    manager.release("first")
    assert manager.refcount(first[0]) == 1
    second = manager.allocate_with_prefix("second", 4, entry.block_ids)
    assert second == first and manager.refcount(first[0]) == 2
    manager.release("second")
    assert manager.refcount(first[0]) == 1
    other = manager.allocate("other", 4)
    cache.insert([9, 8, 7, 6], other)
    assert manager.refcount(first[0]) == 0
    assert manager.free_blocks == 3


def make_runtime():
    config = Qwen3Config(31, 16, 32, 1, 2, 1, 8, max_position_embeddings=64)
    model = NativeQwen3(config, "math", "torch", "torch", "torch").eval()
    manager = BlockManager(32, 4)
    scheduler = Scheduler(manager, max_num_batched_tokens=64)
    cache = model.create_cache(32, 4)
    return model, manager, scheduler, cache


def test_runtime_prefill_then_batched_decode():
    model, manager, scheduler, cache = make_runtime()
    runtime = NativeBatchRuntime(model, scheduler, cache, eos_token_ids=[])
    requests = [
        GenerationRequest("a", [1, 2, 3], max_new_tokens=3),
        GenerationRequest("b", [4, 5, 6], max_new_tokens=3),
    ]
    for request in requests:
        scheduler.add_request(request)
    first = runtime.step()
    assert set(first) == {"a", "b"}
    assert all(request.state is RequestState.RUNNING for request in requests)
    assert all(request.generated_tokens == 1 for request in requests)
    second = runtime.step()
    assert set(second) == {"a", "b"}
    assert all(request.generated_tokens == 2 for request in requests)
    # prompt length 3 + two generated tokens = 5 logical tokens, so each
    # request has grown from one physical page to two pages.
    assert manager.used_blocks == 4


def test_runtime_prefix_hit_keeps_shared_page_after_request_finishes():
    model, manager, scheduler, cache = make_runtime()
    runtime = NativeBatchRuntime(model, scheduler, cache, eos_token_ids=[])
    first = GenerationRequest("a", [1, 2, 3, 4], max_new_tokens=1)
    scheduler.add_request(first)
    runtime.step()
    assert len(runtime.prefix_cache) == 1
    entry = runtime.prefix_cache.lookup(first.prompt_token_ids)
    assert entry is not None
    shared_block = entry.block_ids[0]
    assert manager.refcount(shared_block) == 1
    second = GenerationRequest("b", [1, 2, 3, 4], max_new_tokens=2)
    scheduler.add_request(second)
    runtime.step()
    assert second.prefix_len == 4
    assert second.prefix_block_ids == [shared_block]
    assert manager.refcount(shared_block) >= 2
