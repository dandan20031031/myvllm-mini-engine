import importlib.util
from pathlib import Path

import pytest
import torch

from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config
import mini_engine.model.native_qwen3 as model_ops

spec = importlib.util.spec_from_file_location("engine_profile_entry", Path(__file__).parents[1] / "benchmarks" / "profile_native_batch.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def make_runtime():
    torch.manual_seed(281)
    model = NativeQwen3(Qwen3Config(31, 32, 64, 1, 4, 2, 16), "math").eval()
    runner = NativeQwen3Runner(model, None, num_blocks=16, block_size=4, eos_token_ids=[])
    return runner.create_batch_runtime(enable_prefix_cache=False, prefill_chunk_size=4)


def test_annotations_restore_methods_and_preserve_outputs():
    runtime = make_runtime()
    before = entry.generate(runtime, [[1, 2, 3, 4, 5]], 3)
    original_op = model_ops.paged_attention
    original_forward = runtime.model.lm_head.forward.__func__
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        with entry.instrument_runtime(runtime):
            during = entry.generate(runtime, [[1, 2, 3, 4, 5]], 3, markers=True)
            assert model_ops.paged_attention is not original_op
    assert during["output_sha256"] == before["output_sha256"]
    assert during["output_tokens"] == 3
    assert model_ops.paged_attention is original_op
    assert runtime.model.lm_head.forward.__func__ is original_forward
    assert "forward" not in vars(runtime.model.lm_head)
    assert "decode_step" not in vars(runtime)
    keys = {event.key for event in profile.key_averages()}
    assert {"engine/schedule", "engine/prefill", "engine/decode", "engine/sampling", "op/linear"} <= keys
    assert runtime.scheduler.block_manager.used_blocks == 0


def test_annotations_restore_on_error_and_request_pages_are_released(monkeypatch):
    runtime = make_runtime()
    original = model_ops.paged_attention
    with pytest.raises(RuntimeError, match="injected"):
        with entry.instrument_runtime(runtime, record=False, layers=True):
            def fail(*args, **kwargs):
                raise RuntimeError("injected profiler failure")
            monkeypatch.setattr(runtime.model, "forward", fail)
            entry.generate(runtime, [[1, 2, 3]], 3)
    assert model_ops.paged_attention is original
    assert "schedule" not in vars(runtime.scheduler)
    assert runtime.scheduler.block_manager.used_blocks == 0
    assert not runtime.scheduler.waiting and not runtime.scheduler.running
