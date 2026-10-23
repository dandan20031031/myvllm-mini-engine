"""Single-request greedy prompt-lookup speculation without a second model.

Only the accepted prefix is committed logically. Rejected KV suffixes are
overwritten on the next forward; allocator tail pages are returned immediately.
The target verifies one pending token plus a draft in a single causal forward.
"""

from collections import deque
from dataclasses import dataclass
import time

import torch

from .model.qwen3 import IncrementalGenerationResult


def prompt_lookup(tokens, max_tokens=4, *, min_ngram_size=2, max_ngram_size=4):
    """Copy a known continuation of the longest matching historical suffix.

    Search the complete prompt plus accepted output, preferring the most recent
    earlier occurrence. Every returned token already exists in that history;
    proposals still require target-model verification.
    """
    if max_tokens <= 0:
        return []
    if min_ngram_size <= 0 or max_ngram_size < min_ngram_size:
        raise ValueError("ngram sizes must satisfy 0 < min <= max")
    history = list(tokens)
    for size in range(min(max_ngram_size, len(history) - 1), min_ngram_size - 1, -1):
        suffix = history[-size:]
        for start in range(len(history) - size - 1, -1, -1):
            if history[start:start + size] == suffix:
                return history[start + size:start + size + max_tokens]
    return []


class AcceptanceBudget:
    """Adjust a draft cap using recent accepted/proposed counts, not confidence."""

    def __init__(self, maximum, *, adaptive=False, window=8):
        self.maximum = maximum
        self.adaptive = adaptive
        self.history = deque(maxlen=window)

    @property
    def current(self):
        if not self.adaptive or not self.history or self.maximum == 0:
            return self.maximum
        accepted = sum(pair[0] for pair in self.history)
        proposed = sum(pair[1] for pair in self.history)
        return max(1, min(self.maximum, round(self.maximum * accepted / proposed)))

    def update(self, accepted, proposed):
        if proposed:
            self.history.append((accepted, proposed))


@dataclass
class SpeculativeGenerationResult(IncrementalGenerationResult):
    stats: dict


@torch.inference_mode()
def generate_speculative(
    runner, prompt=None, max_new_tokens=128, *, messages=None,
    max_draft_tokens=4, min_ngram_size=2, max_ngram_size=4,
    adaptive_draft=False, temperature=0.0, top_k=0, top_p=1.0,
):
    """Generate exact greedy target decisions with verified prompt-lookup drafts.

    This path uses eager multi-token suffix verification, not decode CUDA Graph.
    Floating-point differences between suffix and one-token kernels can change
    near-tied argmax decisions; benchmark output equality explicitly.
    """
    if not isinstance(max_new_tokens, int) or isinstance(max_new_tokens, bool) or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")
    if not isinstance(max_draft_tokens, int) or isinstance(max_draft_tokens, bool) or max_draft_tokens < 0:
        raise ValueError("max_draft_tokens must be a nonnegative integer")
    if (not isinstance(min_ngram_size, int) or not isinstance(max_ngram_size, int)
            or min_ngram_size <= 0 or max_ngram_size < min_ngram_size):
        raise ValueError("ngram sizes must satisfy 0 < min <= max")
    if temperature != 0.0 or top_k != 0 or top_p != 1.0:
        raise ValueError("prompt-lookup speculation currently supports greedy decoding only")
    if runner._busy:
        raise RuntimeError("native runner currently supports one active request")
    if messages is None:
        if prompt is None:
            raise ValueError("prompt or messages is required")
        messages = [{"role": "user", "content": prompt}]
    ids = runner.tokenizer.apply_chat_template(
        list(messages), tokenize=True, add_generation_prompt=True,
        enable_thinking=runner.enable_thinking, return_tensors="pt",
    )[0].to(device=runner.cache.device, dtype=torch.long)
    if ids.numel() == 0:
        raise ValueError("prompt must contain at least one token")
    required = ids.numel() + max_new_tokens - 1
    if required > runner.model.config.max_position_embeddings:
        raise ValueError("prompt and output budget exceed context limit")
    if not runner.block_manager.can_allocate(required):
        raise MemoryError("prompt and output budget exceed KV cache capacity")
    eos = runner.eos_token_ids
    if eos is None:
        eos = runner.tokenizer.eos_token_id
    eos_ids = {eos} if isinstance(eos, int) else set(eos or [])
    budget = AcceptanceBudget(max_draft_tokens, adaptive=adaptive_draft)
    history = ids.tolist()
    generated = []
    stats = {
        "method": "greedy_prompt_lookup", "draft_model": None,
        "max_draft_tokens": max_draft_tokens, "adaptive_draft": adaptive_draft,
        "min_ngram_size": min_ngram_size, "max_ngram_size": max_ngram_size,
        "target_forward_calls": 0, "decode_rounds": 0, "verification_rounds": 0,
        "fallback_rounds": 0, "proposed_tokens": 0, "accepted_tokens": 0,
        "rejection_rounds": 0, "bonus_tokens": 0, "rollback_tokens": 0,
        "draft_budgets": [],
    }
    allocated = False
    runner._busy = True
    try:
        blocks = runner.block_manager.allocate("single", ids.numel())
        allocated = True
        runner._sync()
        started = time.perf_counter()
        logits = runner.model(ids, runner.cache, blocks)[-1:]
        stats["target_forward_calls"] += 1
        first_token = int(logits.argmax(-1).item())
        generated.append(first_token)
        history.append(first_token)
        runner._sync()
        first_emitted = last_emitted = time.perf_counter()
        # All history except its final emitted token is present in KV.
        position = ids.numel()
        reason = "stop" if first_token in eos_ids else "length"
        while len(generated) < max_new_tokens and generated[-1] not in eos_ids:
            remaining = max_new_tokens - len(generated)
            cap = min(budget.current, remaining - 1)
            stats["draft_budgets"].append(cap)
            draft = prompt_lookup(
                history, cap, min_ngram_size=min_ngram_size, max_ngram_size=max_ngram_size,
            )
            # Never propose tokens after a known EOS in the copied continuation.
            for index, token_id in enumerate(draft):
                if token_id in eos_ids:
                    draft = draft[:index + 1]
                    break
            stats["proposed_tokens"] += len(draft)
            stats["decode_rounds"] += 1
            stats["verification_rounds" if draft else "fallback_rounds"] += 1
            inputs = torch.tensor([generated[-1], *draft], device=runner.cache.device, dtype=torch.long)
            written_end = position + len(inputs)
            blocks = runner.block_manager.grow("single", written_end)
            verified = runner.model(inputs, runner.cache, blocks, position=position)
            stats["target_forward_calls"] += 1
            if verified.ndim != 2 or verified.shape[0] != len(inputs):
                raise RuntimeError("target verification must return one logit row per input token")
            targets = verified.argmax(-1).tolist()
            accepted = 0
            emitted = []
            for index, draft_token in enumerate(draft):
                if draft_token != targets[index]:
                    emitted.append(int(targets[index]))
                    stats["rejection_rounds"] += 1
                    break
                emitted.append(draft_token)
                accepted += 1
                if draft_token in eos_ids:
                    break
            else:
                # Also handles an empty proposal: ordinary one-token decode.
                emitted.append(int(targets[-1]))
                if draft:
                    stats["bonus_tokens"] += 1
            # Commit the pending input and accepted drafts only. A correction
            # or bonus is the next pending token and is not yet represented in KV.
            keep_tokens = position + 1 + accepted
            stats["rollback_tokens"] += written_end - keep_tokens
            if keep_tokens < written_end:
                blocks = runner.block_manager.truncate("single", keep_tokens)
            position = keep_tokens
            stats["accepted_tokens"] += accepted
            budget.update(accepted, len(draft))
            generated.extend(emitted)
            history.extend(emitted)
            runner._sync()
            last_emitted = time.perf_counter()
            if generated[-1] in eos_ids:
                reason = "stop"
                break
        stats["proposal_hit_rate"] = stats["verification_rounds"] / stats["decode_rounds"] if stats["decode_rounds"] else 0.0
        stats["acceptance_rate"] = stats["accepted_tokens"] / stats["proposed_tokens"] if stats["proposed_tokens"] else 0.0
        stats["output_tokens"] = len(generated)
        stats["tokens_per_target_forward"] = len(generated) / stats["target_forward_calls"]
        elapsed = (last_emitted - started) * 1000.0
        return SpeculativeGenerationResult(
            text=runner.tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=generated, finish_reason=reason,
            ttft_ms=(first_emitted - started) * 1000.0,
            tpot_ms=(last_emitted - first_emitted) * 1000.0 / (len(generated) - 1) if len(generated) > 1 else 0.0,
            total_latency_ms=elapsed, stats=stats,
        )
    finally:
        if allocated:
            runner.block_manager.release("single")
        runner._busy = False
