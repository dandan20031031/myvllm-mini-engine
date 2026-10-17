from types import SimpleNamespace

import pytest

from mini_engine import EngineConfig, LLMEngine


class FakeRunner:
    def generate_incremental(self, prompt, max_new_tokens, **kwargs):
        return SimpleNamespace(
            text=f"echo:{prompt}",
            token_ids=[7],
            finish_reason="stop",
            ttft_ms=1.0,
            tpot_ms=0.0,
            total_latency_ms=1.0,
        )


def test_engine_delegates_incremental_generation():
    engine = LLMEngine(EngineConfig(device="cpu"), model_runner=FakeRunner())
    result = engine.generate_incremental("hello", max_new_tokens=4)
    assert result.text == "echo:hello"
    assert result.token_ids == [7]


def test_incremental_requires_runner():
    engine = LLMEngine(EngineConfig(device="cpu"))
    with pytest.raises(RuntimeError, match="incremental-capable"):
        engine.generate_incremental("hello")
