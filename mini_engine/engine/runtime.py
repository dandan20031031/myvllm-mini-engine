"""Scheduler-driven native batch runtime for prefill/decode experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..cache import PrefixCache
from ..sampling import sample_next_token
from .request import GenerationRequest, RequestState


@dataclass(frozen=True)
class BatchStep:
    prefill: tuple[GenerationRequest, ...]
    decode: tuple[GenerationRequest, ...]


class NativeBatchRuntime:
    """Connect a Scheduler to a NativeQwen3 model.

    Prefill is intentionally one request at a time until a chunked-prefill
    kernel exists. Decode is genuinely batched and uses one model forward for
    all active requests. PrefixCache shares complete KV blocks and protects
    them with allocator reference counts; execution still computes a request's
    full prompt until cached next_logits integration is added.
    """

    def __init__(self, model, scheduler, cache, *, prefix_cache=None, eos_token_ids=None):
        self.model = model
        self.scheduler = scheduler
        self.cache = cache
        self.prefix_cache = prefix_cache or PrefixCache(scheduler.block_manager)
        self.eos_token_ids = set(eos_token_ids or [])

    def prepare_waiting(self) -> None:
        for request in list(self.scheduler.waiting):
            self.admit_prefix(request)

    def admit_prefix(self, request: GenerationRequest) -> int:
        entry = self.prefix_cache.lookup(request.prompt_token_ids)
        if entry is None or entry.next_logits is None:
            return 0
        request.prefix_cache_key = entry.token_key
        request.prefix_block_ids = list(entry.block_ids)
        request.prefix_len = len(entry.token_key)
        return request.prefix_len

    @torch.inference_mode()
    def decode_step(self, requests: list[GenerationRequest]) -> dict[str, int]:
        if not requests:
            return {}
        device = self.cache.device
        tokens = torch.tensor(
            [request.output_token_ids[-1] for request in requests], device=device, dtype=torch.long
        )
        positions = torch.tensor(
            [request.total_len - 1 for request in requests], device=device, dtype=torch.long
        )
        width = max(len(request.block_ids) for request in requests)
        tables = torch.full((len(requests), width), -1, device=device, dtype=torch.long)
        for row, request in enumerate(requests):
            tables[row, : len(request.block_ids)] = torch.tensor(request.block_ids, device=device)
        logits = self.model.forward_decode_batch(tokens, self.cache, tables, positions)
        chosen = sample_next_token(logits, temperature=0.0)
        result = {}
        for row, request in enumerate(requests):
            token_id = int(chosen[row].item())
            finished = token_id in self.eos_token_ids
            self.scheduler.on_token(request, token_id, finished=finished)
            result[request.request_id] = token_id
        return result

    @torch.inference_mode()
    def prefill_step(self, requests: list[GenerationRequest]) -> dict[str, int]:
        """Execute admitted prompts and sample one token per request.

        Prefill is intentionally sequential in this stage; the decode path is
        the first genuinely batched execution path. This makes request/block
        lifecycle correctness testable before chunked-prefill optimization.
        """
        result: dict[str, int] = {}
        for request in requests:
            prompt = torch.tensor(request.prompt_token_ids, device=self.cache.device, dtype=torch.long)
            logits = self.model(prompt, self.cache, request.block_ids)[-1:]
            self.prefix_cache.insert(
                request.prompt_token_ids, request.block_ids, next_logits=logits
            )
            token = sample_next_token(
                logits, request.temperature, request.top_k, request.top_p
            )
            token_id = int(token.item())
            self.scheduler.on_token(request, token_id, finished=token_id in self.eos_token_ids)
            result[request.request_id] = token_id
        return result

    def step(self) -> dict[str, int]:
        """Prepare prefix hits, schedule, prefill new requests, batch decode old ones."""
        self.prepare_waiting()
        output = self.scheduler.schedule()
        generated = self.prefill_step(list(output.prefill))
        active = [request for request in output.decode if not request.is_finished]
        generated.update(self.decode_step(active))
        return generated

    def finish_request(self, request: GenerationRequest) -> None:
        if request.state is RequestState.RUNNING:
            request.finish("stop")
            self.scheduler.running.pop(request.request_id, None)
            self.scheduler.block_manager.release(request.request_id)
