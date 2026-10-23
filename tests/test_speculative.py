"""Prompt-lookup acceptance, rollback and resource ownership without a GPU."""

from types import SimpleNamespace

import pytest
import torch

from mini_engine.model import NativeQwen3, NativeQwen3Runner, Qwen3Config
from mini_engine.speculative import AcceptanceBudget, generate_speculative, prompt_lookup


class Tokenizer:
    eos_token_id = None

    def __init__(self, tokens):
        self.tokens = tokens

    def apply_chat_template(self, *args, **kwargs):
        return torch.tensor([self.tokens], dtype=torch.long)

    def decode(self, tokens, **kwargs):
        return str(tokens)


class TokenModel(torch.nn.Module):
    """Physical-page token cache; every logit depends on the causal prefix."""

    decode_backend = "torch"

    def __init__(self, prediction=None, vocab=17, max_context=128):
        super().__init__()
        self.config = SimpleNamespace(max_position_embeddings=max_context)
        self.prediction = prediction or (lambda prefix: (prefix[-1] + 1) % vocab)
        self.vocab = vocab
        self.calls = []
        self.fail_on_call = None

    def create_cache(self, num_blocks, block_size):
        return SimpleNamespace(device=torch.device("cpu"), block_size=block_size, tokens=[-1] * (num_blocks * block_size))

    def forward(self, ids, cache, blocks, position=0):
        self.calls.append((position, ids.tolist(), list(blocks)))
        if self.fail_on_call == len(self.calls):
            raise RuntimeError("injected target failure")
        rows = []
        for offset, token in enumerate(ids.tolist()):
            absolute = position + offset
            cache.tokens[blocks[absolute // cache.block_size] * cache.block_size + absolute % cache.block_size] = token
            prefix = [cache.tokens[blocks[index // cache.block_size] * cache.block_size + index % cache.block_size] for index in range(absolute + 1)]
            assert -1 not in prefix, "read an uninitialized KV position"
            row = torch.full((self.vocab,), -100.0)
            row[self.prediction(prefix)] = 100.0
            rows.append(row)
        return torch.stack(rows)


def make_runner(tokens=None, prediction=None, eos=None, blocks=32, block_size=2):
    return NativeQwen3Runner(
        TokenModel(prediction), Tokenizer(tokens or [0, 1, 2, 3, 4, 0]),
        num_blocks=blocks, block_size=block_size, eos_token_ids=eos,
    )


def assert_clean(runner):
    assert not runner._busy
    assert runner.block_manager.free_blocks == runner.block_manager.num_blocks


def test_lookup_longest_match_recent_tie_and_no_invented_continuation():
    assert prompt_lookup([1, 2, 3, 8, 1, 2, 4, 9, 1, 2], 2) == [4, 9]
    assert prompt_lookup([1, 2, 3, 4, 9, 2, 3, 8, 1, 2, 3], 2, max_ngram_size=3) == [4, 9]
    assert prompt_lookup([7, 8, 9], 4) == []
    assert prompt_lookup([1, 1, 1], 4, min_ngram_size=1) == [1]
    assert prompt_lookup([1, 2, 1, 2], 0) == []
    with pytest.raises(ValueError):
        prompt_lookup([1, 2], min_ngram_size=0)


def test_full_acceptance_bonus_and_fewer_target_calls():
    runner = make_runner(tokens=list(range(17)) + [0])
    expected = runner.generate_incremental("repeat", 10)
    runner.model.calls.clear()
    actual = generate_speculative(runner, "repeat", 10, max_draft_tokens=4, min_ngram_size=1)
    assert actual.token_ids == expected.token_ids
    assert actual.stats["accepted_tokens"] > 0
    assert actual.stats["bonus_tokens"] > 0
    assert any(len(ids) > 1 for _, ids, _ in runner.model.calls)
    assert actual.stats["target_forward_calls"] == len(runner.model.calls) < len(actual.token_ids)
    assert actual.stats["acceptance_rate"] == 1.0
    assert actual.total_latency_ms >= actual.ttft_ms >= 0
    assert_clean(runner)


def test_rejection_corrects_token_rolls_back_position_and_releases_tail_pages(monkeypatch):
    runner = make_runner()
    expected = runner.generate_incremental("reject", 9).token_ids
    runner.model.calls.clear()
    monkeypatch.setattr("mini_engine.speculative.prompt_lookup", lambda history, cap, **kwargs: [16] * cap)
    actual = generate_speculative(runner, "reject", 9, max_draft_tokens=4)
    assert actual.token_ids == expected
    assert actual.stats["rollback_tokens"] > 0
    assert actual.stats["rejection_rounds"] > 0
    assert actual.stats["accepted_tokens"] == 0
    assert [position for position, _, _ in runner.model.calls[1:]] == list(range(6, 14))
    assert_clean(runner)


def test_accept_prefix_then_correct_and_overwrite_stale_kv(monkeypatch):
    runner = make_runner(prediction=lambda prefix: (sum(prefix) + len(prefix)) % 17)
    expected = runner.generate_incremental("partial", 9).token_ids
    runner.model.calls.clear()
    # Propose a known-correct first token, then deliberately reject a later one.
    def draft(history, cap, **kwargs):
        if not cap:
            return []
        first = (sum(history) + len(history)) % 17
        second = (sum(history) + first + len(history) + 1) % 17
        return [first] + [(second + 1) % 17] * (cap - 1)
    monkeypatch.setattr("mini_engine.speculative.prompt_lookup", draft)
    actual = generate_speculative(runner, "partial", 9, max_draft_tokens=3)
    assert actual.token_ids == expected
    assert actual.stats["accepted_tokens"] > 0
    assert actual.stats["rollback_tokens"] > 0
    assert_clean(runner)


@pytest.mark.parametrize("draft_limit", [0, 4])
def test_no_draft_falls_back_to_ordinary_greedy(draft_limit):
    runner = make_runner(tokens=[0, 7, 11], prediction=lambda prefix: (sum(prefix) + len(prefix)) % 17)
    expected = runner.generate_incremental("none", 6).token_ids
    actual = generate_speculative(runner, "none", 6, max_draft_tokens=draft_limit, min_ngram_size=20, max_ngram_size=20)
    assert actual.token_ids == expected
    assert actual.stats["proposed_tokens"] == 0
    assert actual.stats["fallback_rounds"] == 5
    assert actual.stats["target_forward_calls"] == 6
    assert_clean(runner)


@pytest.mark.parametrize("eos", [1, 2, 4])
def test_eos_at_prefill_accepted_draft_or_bonus(monkeypatch, eos):
    runner = make_runner(tokens=[0], eos=[eos])
    monkeypatch.setattr("mini_engine.speculative.prompt_lookup", lambda history, cap, **kwargs: list(range(history[-1] + 1, history[-1] + 1 + cap)))
    actual = generate_speculative(runner, "eos", 12, max_draft_tokens=2)
    assert actual.token_ids == list(range(1, eos + 1))
    assert actual.finish_reason == "stop"
    if eos == 1:
        assert actual.stats["target_forward_calls"] == 1
    if eos == 2:
        assert actual.stats["accepted_tokens"] == 1
    if eos == 4:
        assert actual.stats["bonus_tokens"] == 1
    assert_clean(runner)


def test_correction_can_be_eos(monkeypatch):
    runner = make_runner(tokens=[0], eos=[2])
    monkeypatch.setattr("mini_engine.speculative.prompt_lookup", lambda history, cap, **kwargs: [9] * cap)
    result = generate_speculative(runner, "correction", 10)
    assert result.token_ids == [1, 2] and result.finish_reason == "stop"
    assert result.stats["accepted_tokens"] == 0
    assert_clean(runner)


@pytest.mark.parametrize("max_new_tokens", [1, 2, 3, 7])
def test_length_budget_never_writes_past_reserved_context(max_new_tokens):
    runner = make_runner(tokens=[0, 1, 2, 0], blocks=4, block_size=4)
    actual = generate_speculative(runner, "length", max_new_tokens, min_ngram_size=1, max_draft_tokens=20)
    assert len(actual.token_ids) == max_new_tokens
    assert actual.finish_reason == "length"
    assert all(position + len(ids) <= 4 + max_new_tokens - 1 for position, ids, _ in runner.model.calls)
    assert_clean(runner)


def test_exception_cleanup_and_subsequent_request(monkeypatch):
    runner = make_runner()
    runner.model.fail_on_call = 2
    with pytest.raises(RuntimeError, match="injected"):
        generate_speculative(runner, "failure", 8, min_ngram_size=1)
    assert_clean(runner)
    runner.model.fail_on_call = None
    assert len(generate_speculative(runner, "next", 3).token_ids) == 3
    assert_clean(runner)


def test_owned_single_pages_are_not_released_when_allocation_fails():
    runner = make_runner()
    owned = runner.block_manager.allocate("single", 2)
    with pytest.raises(ValueError, match="already owns"):
        generate_speculative(runner, "collision", 3)
    assert list(runner.block_manager.table("single").block_ids) == owned
    assert not runner._busy
    runner.block_manager.release("single")


def test_parameter_and_capacity_guards():
    runner = make_runner(blocks=4)
    for kwargs in ({"max_new_tokens": 0}, {"max_draft_tokens": -1}, {"min_ngram_size": 0}, {"temperature": 0.5}, {"top_p": 0.9}):
        with pytest.raises(ValueError):
            generate_speculative(runner, "bad", **kwargs)
    with pytest.raises(MemoryError):
        generate_speculative(runner, "capacity", 20)
    runner._busy = True
    with pytest.raises(RuntimeError, match="active request"):
        generate_speculative(runner, "busy", 2)
    runner._busy = False
    assert_clean(runner)


def test_adaptive_budget_uses_recent_acceptance_rate():
    budget = AcceptanceBudget(4, adaptive=True)
    assert budget.current == 4
    budget.update(0, 4)
    assert budget.current == 1
    for _ in range(8):
        budget.update(4, 4)
    assert budget.current == 4
    fixed = AcceptanceBudget(4, adaptive=False)
    fixed.update(0, 4)
    assert fixed.current == 4


def test_tiny_native_model_matches_one_token_greedy():
    torch.manual_seed(41)
    cfg = Qwen3Config(17, 16, 32, 1, 2, 1, 8, max_position_embeddings=64)
    runner = NativeQwen3Runner(NativeQwen3(cfg, "math"), Tokenizer([1, 3, 1, 3]), num_blocks=8, block_size=4)
    expected = runner.generate_incremental("tiny", 8)
    actual = generate_speculative(runner, "tiny", 8, min_ngram_size=1, adaptive_draft=True)
    assert actual.token_ids == expected.token_ids
    assert actual.stats["verification_rounds"] > 0
    assert_clean(runner)


@pytest.mark.parametrize("seed", range(8))
def test_causal_prefix_property_across_rejections_lengths_and_eos(seed):
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(0, 17, (10,), generator=generator).tolist()
    prediction = lambda prefix: (sum((index + 1) * value for index, value in enumerate(prefix)) + len(prefix) * seed) % 17
    runner = make_runner(tokens, prediction, eos=[seed] if seed % 2 else None)
    expected = runner.generate_incremental("property", 20)
    result = generate_speculative(
        runner, "property", 20, max_draft_tokens=6,
        min_ngram_size=1, adaptive_draft=bool(seed % 2),
    )
    assert result.token_ids == expected.token_ids
    assert result.finish_reason == expected.finish_reason
    assert_clean(runner)
