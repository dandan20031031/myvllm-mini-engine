"""Startup parity, early rejection and cleanup without loading model weights."""

import argparse

import pytest
import torch

from mini_engine import RunnerConfig
from mini_engine.cli import add_runner_arguments, runner_config_from_args
from mini_engine.model import NativeQwen3Runner


def test_chat_and_http_share_model_settings_and_public_runner_arguments():
    options = [
        "--model", "local-model", "--dtype", "fp16", "--num-blocks", "64",
        "--prefill-backend", "dandan", "--decode-backend", "cuda",
        "--data-backend", "cuda", "--elementwise-backend", "cuda",
        "--linear-backend", "dandan_cuda", "--sampling-backend", "cuda",
        "--quantization", "int4", "--quant-backend", "dandan", "--quant-group-size", "64",
        "--local-files-only", "--fuse-projections", "--cuda-graph",
    ]
    configs = []
    for serving in (False, True):
        parser = argparse.ArgumentParser()
        add_runner_arguments(parser, serving=serving)
        configs.append(runner_config_from_args(parser.parse_args(options)))
    assert configs[0] == configs[1]
    assert isinstance(configs[0], RunnerConfig)
    kwargs = configs[0].runner_kwargs()
    assert kwargs["dtype"] is torch.float16
    assert kwargs["quantization"].activation_dtype == "fp16"
    assert kwargs["quantization"].group_size == 64
    assert kwargs["use_cuda_graph"] and kwargs["local_files_only"]
    # Every translated option is accepted by the existing public API.
    import inspect
    inspect.signature(NativeQwen3Runner.from_pretrained).bind("local-model", **kwargs)


@pytest.mark.parametrize("entry_point", ["chat", "server"])
@pytest.mark.parametrize("invalid_options", [
    ["--cuda-graph"],
    ["--device", "cpu", "--decode-backend", "cuda"],
    ["--dtype", "fp32", "--data-backend", "cuda"],
    ["--linear-backend", "dandan"],
    ["--quant-backend", "dandan"],
    ["--num-blocks", "0"],
    ["--max-new-tokens", "0"],
])
def test_invalid_startup_fails_before_loading(monkeypatch, entry_point, invalid_options):
    from mini_engine.__main__ import main as chat_main
    from mini_engine.server import main as server_main

    def unexpected(*args, **kwargs):
        pytest.fail("invalid startup reached model loading")

    monkeypatch.setattr(NativeQwen3Runner, "from_pretrained", unexpected)
    argv = ["mini_engine", "--model", "not-downloaded"]
    if entry_point == "chat":
        argv += ["--prompt", "hello"]
    monkeypatch.setattr("sys.argv", argv + invalid_options)
    with pytest.raises(SystemExit) as error:
        (chat_main if entry_point == "chat" else server_main)()
    assert error.value.code == 2


@pytest.mark.parametrize("failure", ["construct", "start", "bind", None])
def test_server_startup_and_shutdown_release_resources_in_order(monkeypatch, failure):
    import mini_engine.server as entry

    events = []

    class Runner:
        def close(self):
            events.append("runner.close")

    class Service:
        def __init__(self, *args, **kwargs):
            events.append("service.construct")
            if failure == "construct":
                raise RuntimeError("construct failed")

        def start(self):
            events.append("service.start")
            if failure == "start":
                raise RuntimeError("start failed")
            return self

        def close(self):
            events.append("service.close")

    class Server:
        server_port = 8000

        def __init__(self, *args, **kwargs):
            events.append("server.bind")
            if failure == "bind":
                raise RuntimeError("bind failed")

        def serve_forever(self, **kwargs):
            events.append("server.serve")
            raise KeyboardInterrupt

        def server_close(self):
            events.append("server.close")

    monkeypatch.setattr(NativeQwen3Runner, "from_pretrained", lambda *args, **kwargs: Runner())
    monkeypatch.setattr(entry, "InferenceService", Service)
    monkeypatch.setattr(entry, "ChatHTTPServer", Server)
    monkeypatch.setattr("sys.argv", ["mini_engine.server", "--model", "local-model"])
    if failure is not None:
        with pytest.raises(RuntimeError, match=failure):
            entry.main()
    else:
        entry.main()
    assert events[-1] == "runner.close"
    if failure != "construct":
        assert events[-2] == "service.close"
    if failure is None:
        assert events[-3] == "server.close"


def test_server_does_not_close_runner_while_worker_join_has_failed(monkeypatch):
    import mini_engine.server as entry

    events = []

    class Service:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            raise RuntimeError("worker startup timeout")

        def close(self):
            events.append("service.close")
            raise RuntimeError("worker is still executing")

    class Runner:
        def close(self):
            events.append("runner.close")

    monkeypatch.setattr(NativeQwen3Runner, "from_pretrained", lambda *args, **kwargs: Runner())
    monkeypatch.setattr(entry, "InferenceService", Service)
    monkeypatch.setattr("sys.argv", ["mini_engine.server", "--model", "local-model"])
    with pytest.raises(RuntimeError, match="still executing"):
        entry.main()
    assert events == ["service.close"]


def test_generation_result_compatibility_imports():
    from mini_engine.model import IncrementalGenerationResult
    from mini_engine.model.outputs import IncrementalGenerationResult as SharedResult
    from mini_engine.model.qwen3 import IncrementalGenerationResult as ReferenceResult

    assert IncrementalGenerationResult is SharedResult is ReferenceResult
