"""Public chat/CLI graph selection and lifecycle regression checks."""

import copy
import os
from types import SimpleNamespace

import pytest
import torch

from mini_engine.__main__ import main
from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config
from mini_engine.model.qwen3 import IncrementalGenerationResult


class ChatTokenizer:
    eos_token_id = None

    def __init__(self, ids=(1, 3, 5)):
        self.ids = tuple(ids)

    def apply_chat_template(self, messages, **kwargs):
        return torch.tensor([self.ids], dtype=torch.long)

    def decode(self, ids, **kwargs):
        return " ".join(map(str, ids))


def install_cli_runner(monkeypatch):
    calls = {"load": [], "chat": [], "generate": [], "close": 0}

    class FakeRunner:
        model = SimpleNamespace(quantization_report=None, backend_report=lambda: {})

        def chat(self, messages, maximum, **kwargs):
            calls["chat"].append((copy.deepcopy(messages), maximum, kwargs))
            return "answer"

        def generate_incremental(self, prompt, maximum, **kwargs):
            calls["generate"].append((prompt, maximum, kwargs))
            return IncrementalGenerationResult("answer", [1], "length", 1., 0., 1.)

        def close(self):
            calls["close"] += 1

    def load(cls, *args, **kwargs):
        calls["load"].append((args, kwargs))
        return FakeRunner()

    monkeypatch.setattr(NativeQwen3Runner, "from_pretrained", classmethod(load))
    return calls


def test_cli_cuda_graph_routes_each_chat_turn_and_clear_keeps_history_correct(monkeypatch, capsys):
    calls = install_cli_runner(monkeypatch)
    commands = iter(("first", "second", "/clear", "new conversation", "/exit"))
    monkeypatch.setattr("builtins.input", lambda prompt: next(commands))
    monkeypatch.setattr("sys.argv", ["mini_engine", "--chat", "--cuda-graph", "--max-new-tokens", "4",
                                    "--decode-backend", "cuda", "--data-backend", "cuda"])
    main()
    assert len(calls["load"]) == 1
    assert [len(messages) for messages, _, _ in calls["chat"]] == [1, 3, 1]
    assert all(kwargs.get("use_cuda_graph") is True for _, _, kwargs in calls["chat"])
    assert calls["chat"][2][0] == [{"role": "user", "content": "new conversation"}]
    assert "conversation cleared" in capsys.readouterr().out


def test_cli_default_chat_remains_eager(monkeypatch):
    calls = install_cli_runner(monkeypatch)
    commands = iter(("hello", "/exit"))
    monkeypatch.setattr("builtins.input", lambda prompt: next(commands))
    monkeypatch.setattr("sys.argv", ["mini_engine", "--chat"])
    main()
    assert calls["chat"][0][2].get("use_cuda_graph", False) is False


def test_cli_single_prompt_forwards_cuda_graph_flag(monkeypatch, capsys):
    calls = install_cli_runner(monkeypatch)
    monkeypatch.setattr("sys.argv", ["mini_engine", "--prompt", "hello", "--cuda-graph", "--max-new-tokens", "4",
                                    "--decode-backend", "cuda", "--data-backend", "cuda"])
    main()
    assert calls["generate"] == [("hello", 4, {"use_cuda_graph": True})]
    assert '"text": "answer"' in capsys.readouterr().out


def test_cli_rejects_graph_with_speculative_before_loading_weights(monkeypatch, capsys):
    calls = install_cli_runner(monkeypatch)
    monkeypatch.setattr("sys.argv", ["mini_engine", "--prompt", "hello", "--cuda-graph", "--speculative", "prompt_lookup"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert not calls["load"]
    assert "speculative" in capsys.readouterr().err.lower()


def make_tiny_runner(*, device="cpu", dtype=torch.float32, ids=(1, 3, 5), num_blocks=24):
    torch.manual_seed(291)
    config = Qwen3Config(41, 32, 64, 2, 4, 2, 16, max_position_embeddings=64)
    cuda = device == "cuda"
    model = NativeQwen3(config, "math", "cuda" if cuda else "torch",
                       "cuda" if cuda else "torch", "cuda" if cuda else "torch",
                       "cuda" if cuda else "torch").to(device=device, dtype=dtype).eval()
    return NativeQwen3Runner(model, ChatTokenizer(ids), num_blocks=num_blocks, block_size=4, eos_token_ids=[])


def test_graph_generation_rejects_cpu_without_allocating_request_pages():
    runner = make_tiny_runner()
    with pytest.raises((ValueError, RuntimeError), match="CUDA"):
        runner.generate_incremental("hello", 4, use_cuda_graph=True)
    assert runner.block_manager.free_blocks == runner.block_manager.num_blocks
    assert not runner._busy


CUDA_CHAT = pytest.mark.skipif(
    os.environ.get("MINI_ENGINE_TEST_CUDA") != "1" or not torch.cuda.is_available(),
    reason="set MINI_ENGINE_TEST_CUDA=1 with CUDA available",
)


@CUDA_CHAT
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_chat_graph_matches_eager_across_pages_turns_and_close(dtype):
    runner = make_tiny_runner(device="cuda", dtype=dtype)
    try:
        for ids in ((1, 3, 5), (5, 2, 9, 1), (2, 4, 6)):
            runner.tokenizer.ids = ids
            eager = runner.generate_incremental("hello", 9, use_cuda_graph=False)
            graph = runner.generate_incremental("hello", 9, use_cuda_graph=True)
            assert graph.token_ids == eager.token_ids
            assert graph.finish_reason == eager.finish_reason
            assert runner.block_manager.used_blocks == 1  # only private scratch
            assert not runner._busy
        assert runner.chat_graph_status()["capture_count"] == 1
        manager = runner.chat_graph_manager
        runner.generate_incremental("hello", 17, use_cuda_graph=True)
        assert runner.chat_graph_manager is manager
        assert runner.chat_graph_status()["capture_count"] == 2
    finally:
        runner.close()
    runner.close()
    assert runner.block_manager.used_blocks == 0
    assert runner.chat_graph_status()["captured_buckets"] == []


@CUDA_CHAT
def test_chat_graph_single_token_and_prefill_eos_need_no_capture():
    runner = make_tiny_runner(device="cuda", dtype=torch.float16)
    try:
        first = runner.generate_incremental("hello", 1, use_cuda_graph=True)
        assert runner.chat_graph_status()["capture_count"] == 0
        runner.eos_token_ids = first.token_ids
        stopped = runner.generate_incremental("hello", 8, use_cuda_graph=True)
        assert stopped.token_ids == first.token_ids and stopped.finish_reason == "stop"
        assert runner.block_manager.used_blocks == 0
        assert runner.chat_graph_status()["capture_count"] == 0
    finally:
        runner.close()


@CUDA_CHAT
def test_chat_graph_capture_failure_and_budget_rejection_release_request(monkeypatch):
    from mini_engine.engine.cuda_graph import DecodeGraphBucket
    runner = make_tiny_runner(device="cuda", dtype=torch.float16, num_blocks=4)
    try:
        with pytest.raises(MemoryError):
            runner.generate_incremental("hello", 14, use_cuda_graph=True)
        assert runner.block_manager.used_blocks == 0
        with pytest.raises(ValueError, match="greedy"):
            runner.generate_incremental("hello", 4, use_cuda_graph=True, temperature=0.7)
        def fail_capture(*args, **kwargs):
            raise RuntimeError("injected capture failure")
        monkeypatch.setattr(DecodeGraphBucket, "capture", fail_capture)
        with pytest.raises(RuntimeError, match="injected capture"):
            runner.generate_incremental("hello", 4, use_cuda_graph=True)
        assert not runner._busy and runner.block_manager.used_blocks == 1
        assert len(runner.generate_incremental("hello", 4, use_cuda_graph=False).token_ids) == 4
    finally:
        runner.close()
    assert runner.block_manager.used_blocks == 0


@CUDA_CHAT
def test_batch_graph_metadata_and_sampling_match_eager_when_active_slots_shrink():
    from mini_engine.engine.request import GenerationRequest

    outputs = []
    for graph in (False, True):
        runner = make_tiny_runner(device="cuda", dtype=torch.float16)
        runtime = runner.create_batch_runtime(
            use_cuda_graph=graph, graph_batch_sizes=(1, 2, 4), graph_block_buckets=(1, 2, 4, 8),
            enable_prefix_cache=False)
        requests = [GenerationRequest(str(i), [1, 3, 5, 7][:i + 1], max_new_tokens=9 - i * 2)
                    for i in range(4)]
        try:
            for request in requests:
                runtime.scheduler.add_request(request)
            for _ in range(16):
                if not (runtime.scheduler.waiting or runtime.scheduler.running):
                    break
                runtime.step()
            assert all(request.is_finished for request in requests)
            outputs.append([request.output_token_ids for request in requests])
            assert runner.block_manager.used_blocks == (4 if graph else 0)
        finally:
            runtime.close()
            runner.close()
        assert runner.block_manager.used_blocks == 0
    assert outputs[0] == outputs[1]
