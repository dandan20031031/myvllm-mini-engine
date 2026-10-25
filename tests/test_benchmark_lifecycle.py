"""Benchmark/profile failures release sessions without loading model weights."""

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mini_engine.engine.request import GenerationRequest
from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config
import mini_engine.model.native_qwen3 as model_ops


def load_entry(name):
    path = Path(__file__).parents[1] / "benchmarks" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name + "_lifecycle", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return torch.tensor([[1, 2, 3]]) if kwargs.get("return_tensors") else [1, 2, 3]


@pytest.fixture
def session(monkeypatch):
    model = NativeQwen3(Qwen3Config(31, 32, 64, 1, 4, 2, 16), "math").eval()
    runner = NativeQwen3Runner(model, Tokenizer(), num_blocks=16, block_size=4, eos_token_ids=[])
    state = SimpleNamespace(runner=runner, runtime=None, events=[])
    create = runner.create_batch_runtime
    close_runner = runner.close

    def create_runtime(**kwargs):
        runtime = create(**kwargs)
        state.runtime = runtime
        close_runtime = runtime.close

        def close():
            if not runtime.closed:
                state.events.append("runtime")
            close_runtime()

        monkeypatch.setattr(runtime, "close", close)
        return runtime

    def close():
        state.events.append("runner")
        close_runner()

    monkeypatch.setattr(runner, "create_batch_runtime", create_runtime)
    monkeypatch.setattr(runner, "close", close)
    monkeypatch.setattr(NativeQwen3Runner, "from_pretrained", classmethod(lambda cls, *a, **k: runner))
    # These tests exercise CPU lifecycle only, including result metadata.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    return state


def batch_args():
    return SimpleNamespace(model="unused", device="cpu", dtype="fp32", prefill_backend="math",
        decode_backend="torch", elementwise_backend="torch", data_backend="torch", linear_backend="torch",
        num_blocks=16, block_size=4, local_files_only=True, fuse_projections=False,
        quantization="none", quant_backend="reference", quant_group_size=128, prefix_cache="disabled",
        ignore_eos=True, prompts=["hello"], warmup=0, repetitions=1, max_new_tokens=3)


def fail_with_live_request(runtime, *args, **kwargs):
    runtime.scheduler.add_request(GenerationRequest("unfinished", [1, 2, 3], max_new_tokens=3))
    runtime.scheduler.schedule()
    assert runtime.scheduler.block_manager.used_blocks > 0
    raise RuntimeError("injected execution failure")


@pytest.mark.parametrize("failure", [None, "create", "encode", "execute"])
def test_native_batch_closes_sessions_on_success_and_each_startup_stage(monkeypatch, session, failure):
    entry = load_entry("benchmark_native_batch")
    if failure == "create":
        def fail_create(**kwargs):
            raise RuntimeError("injected creation failure")
        monkeypatch.setattr(session.runner, "create_batch_runtime", fail_create)
    elif failure == "encode":
        def fail_encode(*args):
            raise RuntimeError("injected tokenization failure")
        monkeypatch.setattr(entry, "encode_prompts", fail_encode)
    elif failure == "execute":
        monkeypatch.setattr(entry, "run_once", fail_with_live_request)
    if failure is None:
        result = entry.run_mode(batch_args(), use_cuda_graph=False)
        assert result["samples"][0]["output_tokens"] == 3
    else:
        with pytest.raises(RuntimeError, match="injected"):
            entry.run_mode(batch_args(), use_cuda_graph=False)
    assert session.events == (["runner"] if failure == "create" else ["runtime", "runner"])
    if session.runtime is not None:
        assert session.runtime.closed
    assert session.runner.block_manager.used_blocks == 0


def test_profiler_restores_instrumentation_before_closing_failed_session(monkeypatch, tmp_path, session):
    entry = load_entry("profile_native_batch")
    original_op = model_ops.paged_attention
    calls = 0

    def generate(runtime, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            return {"wall_ms": 1., "steps": 1, "output_tokens": 1, "output_sha256": "warmup"}
        assert model_ops.paged_attention is not original_op
        fail_with_live_request(runtime)

    create = session.runner.create_batch_runtime

    def create_runtime(**kwargs):
        runtime = create(**kwargs)
        close = runtime.close

        def assert_restored_then_close():
            assert model_ops.paged_attention is original_op
            assert "decode_step" not in vars(runtime)
            assert "schedule" not in vars(runtime.scheduler)
            close()

        monkeypatch.setattr(runtime, "close", assert_restored_then_close)
        return runtime

    monkeypatch.setattr(session.runner, "create_batch_runtime", create_runtime)
    monkeypatch.setattr(entry, "generate", generate)
    monkeypatch.setattr("sys.argv", ["profile", "--device", "cpu", "--dtype", "fp32", "--tool", "nvtx",
        "--prefill-backend", "math", "--linear-backend", "torch", "--prompt-tokens", "3",
        "--max-new-tokens", "3", "--warmup", "1", "--output-dir", str(tmp_path)])
    with pytest.raises(RuntimeError, match="injected"):
        entry.main()
    assert session.events == ["runtime", "runner"]
    assert session.runtime.closed and session.runner.block_manager.used_blocks == 0


def test_system_validator_failure_restores_hook_and_closes_session(monkeypatch, session):
    entry = load_entry("validate_system_extensions")
    args = batch_args()
    args.chunk_size, args.token_budget = 4, 16
    monkeypatch.setattr(entry, "fresh_runner", lambda *a: session.runner)
    monkeypatch.setattr(entry, "execute", fail_with_live_request)
    previous_hooks = dict(session.runner.model._forward_pre_hooks)
    with pytest.raises(RuntimeError, match="injected"):
        entry.validate_chunks(session.runner, args, [1, 2, 3])
    assert session.events == ["runtime", "runner"]
    assert session.runtime.closed and session.runner.block_manager.used_blocks == 0
    assert session.runner.model._forward_pre_hooks == previous_hooks
