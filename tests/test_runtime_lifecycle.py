"""Batch sessions release their own resources without retiring borrowed ones."""

import os
from types import SimpleNamespace

import pytest
import torch

from mini_engine.engine.request import GenerationRequest, RequestState
from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config


class FakeGraphManager:
    def __init__(self, *args, batch_sizes=(1, 2), block_buckets=(2, 4), **kwargs):
        self.batch_sizes = batch_sizes
        self.block_buckets = block_buckets
        self.buckets = {}

    def shape_for(self, batch_size, max_blocks):
        return batch_size, max_blocks

    def get(self, batch_size, max_blocks, *, padding_block_ids):
        key = self.shape_for(batch_size, max_blocks)
        bucket = self.buckets.get(key)
        if bucket is None:
            bucket = SimpleNamespace(padding_block_ids=list(padding_block_ids), captured=False)
            self.buckets[key] = bucket
        elif bucket.padding_block_ids is None:
            bucket.padding_block_ids = list(padding_block_ids)
        return bucket


def make_runner():
    config = Qwen3Config(23, 16, 32, 1, 2, 1, 8, max_position_embeddings=64)
    model = NativeQwen3(config, "math")
    return NativeQwen3Runner(model, None, num_blocks=24, block_size=4, eos_token_ids=[])


def test_close_aborts_waiting_and_running_and_releases_prefix_refs():
    runner = make_runner()
    unrelated = runner.block_manager.allocate("other-owner", 4)
    runtime = runner.create_batch_runtime()
    running = GenerationRequest("running", [1, 2, 3, 4], max_new_tokens=4)
    runtime.scheduler.add_request(running)
    runtime.step()
    waiting = GenerationRequest("waiting", [5, 6], max_new_tokens=2)
    runtime.scheduler.add_request(waiting)
    assert len(runtime.prefix_cache) == 1

    runtime.close()
    runtime.close()

    assert runtime.closed
    assert running.state is waiting.state is RequestState.ABORTED
    assert not runtime.scheduler.running and not runtime.scheduler.waiting
    assert len(runtime.prefix_cache) == 0
    assert runner.block_manager.used_blocks == 1
    assert runner.block_manager.refcount(unrelated[0]) == 1
    # Closing a batch session leaves model weights and the runner usable.
    next_runtime = runner.create_batch_runtime()
    request = GenerationRequest("next", [1, 2], max_new_tokens=1)
    next_runtime.scheduler.add_request(request)
    assert "next" in next_runtime.step()
    next_runtime.close()


@pytest.mark.parametrize("operation", [
    lambda runtime: runtime.step(),
    lambda runtime: runtime.prefill_step([]),
    lambda runtime: runtime.decode_step([]),
    lambda runtime: runtime.prepare_waiting(),
    lambda runtime: runtime.admit_prefix(GenerationRequest("late", [1])),
])
def test_closed_runtime_rejects_execution(operation):
    runtime = make_runner().create_batch_runtime()
    runtime.close()
    with pytest.raises(RuntimeError, match="closed"):
        operation(runtime)


def test_close_restores_scheduler_callbacks_without_overwriting_new_callbacks():
    runtime = make_runner().create_batch_runtime()
    custom_evict = lambda *args, **kwargs: False
    runtime.scheduler.evict_cache = custom_evict
    runtime.close()
    assert runtime.scheduler.prepare_request is None
    assert runtime.scheduler.evict_cache is custom_evict


def test_owned_graph_manager_and_scratch_are_released(monkeypatch):
    monkeypatch.setattr("mini_engine.engine.DecodeGraphManager", FakeGraphManager)
    runner = make_runner()
    runtime = runner.create_batch_runtime(use_cuda_graph=True)
    manager = runtime.graph_manager
    bucket = SimpleNamespace(graph=object(), logits=object(), sampled_token_ids=object())
    manager.buckets["captured"] = bucket
    assert runner.block_manager.used_blocks == 8
    runtime.close()
    assert manager.buckets == {}
    assert bucket.graph is bucket.logits is bucket.sampled_token_ids is None
    assert runtime.graph_manager is None
    assert runtime.graph_padding_block_ids is None
    assert runner.block_manager.used_blocks == 0


def test_borrowed_manager_and_caller_padding_are_preserved():
    runner = make_runner()
    padding = runner.block_manager.allocate("caller-padding", 8)
    manager = FakeGraphManager()
    existing = object()
    manager.buckets["existing"] = existing
    runtime = runner.create_batch_runtime(graph_manager=manager, graph_padding_block_ids=padding)
    bucket = runtime._get_graph_bucket(1, 2, padding)
    runtime.close()
    assert manager.buckets["existing"] is existing
    assert manager.buckets[(1, 2)] is bucket
    assert all(runner.block_manager.refcount(page) == 1 for page in padding)


def test_borrowed_manager_loses_only_session_buckets_when_scratch_is_owned():
    runner = make_runner()
    manager = FakeGraphManager()
    existing = SimpleNamespace(padding_block_ids=[20], captured=True)
    manager.buckets[(1, 4)] = existing
    runtime = runner.create_batch_runtime(graph_manager=manager)
    runtime._get_graph_bucket(1, 2, runtime.graph_padding_block_ids)
    assert (1, 2) in manager.buckets
    runtime.close()
    assert manager.buckets == {(1, 4): existing}
    assert runner.block_manager.used_blocks == 0


def test_uninitialized_borrowed_bucket_is_restored_after_session():
    runner = make_runner()
    manager = FakeGraphManager()
    original = SimpleNamespace(padding_block_ids=None, captured=False)
    manager.buckets[(1, 2)] = original
    runtime = runner.create_batch_runtime(graph_manager=manager)
    replacement = runtime._get_graph_bucket(1, 2, runtime.graph_padding_block_ids)
    assert replacement is not original
    assert original.padding_block_ids is None
    runtime.close()
    assert manager.buckets[(1, 2)] is original
    assert original.padding_block_ids is None


def test_borrowed_bucket_replacement_failure_restores_original(monkeypatch):
    runner = make_runner()
    manager = FakeGraphManager()
    original = SimpleNamespace(padding_block_ids=None, captured=False)
    manager.buckets[(1, 2)] = original
    runtime = runner.create_batch_runtime(graph_manager=manager)

    def fail_get(*args, **kwargs):
        raise RuntimeError("injected bucket construction failure")

    monkeypatch.setattr(manager, "get", fail_get)
    with pytest.raises(RuntimeError, match="bucket construction"):
        runtime._get_graph_bucket(1, 2, runtime.graph_padding_block_ids)
    assert manager.buckets[(1, 2)] is original
    runtime.close()
    assert runner.block_manager.used_blocks == 0


def test_sessions_use_distinct_scratch_owners():
    runner = make_runner()
    first = runner.create_batch_runtime(graph_manager=FakeGraphManager())
    second = runner.create_batch_runtime(graph_manager=FakeGraphManager())
    first_pages = tuple(first.graph_padding_block_ids)
    second_pages = tuple(second.graph_padding_block_ids)
    assert set(first_pages).isdisjoint(second_pages)
    first.close()
    assert all(runner.block_manager.refcount(page) == 0 for page in first_pages)
    assert all(runner.block_manager.refcount(page) == 1 for page in second_pages)
    second.close()
    assert runner.block_manager.used_blocks == 0


@pytest.mark.parametrize("borrowed", [False, True])
def test_factory_failure_rolls_back_only_its_resources(monkeypatch, borrowed):
    runner = make_runner()
    first = runner.create_batch_runtime(graph_manager=FakeGraphManager())
    first_pages = tuple(first.graph_padding_block_ids)
    manager = FakeGraphManager()
    existing = object()
    manager.buckets["caller"] = existing
    created = []

    def make_manager(*args, **kwargs):
        result = FakeGraphManager(*args, **kwargs)
        result.buckets["internal"] = object()
        created.append(result)
        return result

    def fail_runtime(*args, **kwargs):
        raise RuntimeError("injected runtime construction failure")

    monkeypatch.setattr("mini_engine.engine.DecodeGraphManager", make_manager)
    monkeypatch.setattr("mini_engine.engine.NativeBatchRuntime", fail_runtime)
    with pytest.raises(RuntimeError, match="runtime construction"):
        runner.create_batch_runtime(graph_manager=manager if borrowed else None, use_cuda_graph=True)
    assert runner.block_manager.used_blocks == len(first_pages)
    assert all(runner.block_manager.refcount(page) == 1 for page in first_pages)
    assert manager.buckets == {"caller": existing}
    if not borrowed:
        assert created[0].buckets == {}
    first.close()


def test_close_synchronizes_before_recycling_graph_pages(monkeypatch):
    monkeypatch.setattr("mini_engine.engine.DecodeGraphManager", FakeGraphManager)
    runner = make_runner()
    runtime = runner.create_batch_runtime(use_cuda_graph=True)
    runtime.cache = SimpleNamespace(device=torch.device("cuda"))
    manager = runtime.graph_manager
    manager.buckets["captured"] = object()
    observed = []

    def synchronize(device):
        observed.append((device, runner.block_manager.used_blocks, bool(manager.buckets)))

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    runtime.close()
    assert observed == [(torch.device("cuda"), 8, True)]
    assert runner.block_manager.used_blocks == 0
    runtime.close()
    assert len(observed) == 1


def test_failed_cuda_synchronization_retains_resources_for_safe_retry(monkeypatch):
    runner = make_runner()
    runtime = runner.create_batch_runtime(graph_manager=FakeGraphManager())
    runtime.cache = SimpleNamespace(device=torch.device("cuda"))

    def fail_sync(device):
        raise RuntimeError("injected CUDA synchronization failure")

    monkeypatch.setattr(torch.cuda, "synchronize", fail_sync)
    with pytest.raises(RuntimeError, match="synchronization"):
        runtime.close()
    assert not runtime.closed
    assert runner.block_manager.used_blocks == 2
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: None)
    runtime.close()
    assert runtime.closed
    assert runner.block_manager.used_blocks == 0


@pytest.mark.skipif(
    os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set MINI_ENGINE_TEST_CUDA=1 with CUDA available",
)
def test_real_cuda_close_retires_replay_before_kv_pages_are_reused():
    config = Qwen3Config(41, 32, 64, 2, 4, 2, 16, max_position_embeddings=64)
    model = NativeQwen3(config, "math", "cuda", "cuda", "cuda", "cuda").to(
        device="cuda", dtype=torch.float16)
    runner = NativeQwen3Runner(model, None, num_blocks=24, block_size=4, eos_token_ids=[])
    runtime = runner.create_batch_runtime(use_cuda_graph=True, graph_batch_sizes=(1, 2),
        graph_block_buckets=(1, 2, 4, 8), enable_prefix_cache=False)
    request = GenerationRequest("graph", [1, 2, 3], max_new_tokens=4)
    runtime.scheduler.add_request(request)
    runtime.step()
    runtime.step()
    manager = runtime.graph_manager
    bucket = next(iter(manager.buckets.values()))
    assert bucket.captured
    runtime.close()
    assert runner.block_manager.used_blocks == 0
    assert not manager.buckets and not bucket.captured
    with pytest.raises(RuntimeError, match="captured"):
        bucket.replay_metadata([1], [0], [[0]])

    next_runtime = runner.create_batch_runtime(enable_prefix_cache=False)
    next_runtime.scheduler.add_request(GenerationRequest("reuse", [2, 3], max_new_tokens=1))
    assert "reuse" in next_runtime.step()
    next_runtime.close()
    runner.close()
    assert runner.block_manager.used_blocks == 0
