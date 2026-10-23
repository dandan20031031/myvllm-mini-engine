"""Scheduler-driven native batch runtime for prefill/decode experiments."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from ..cache import PrefixCache
from .cuda_graph import DecodeGraphManager
from ..sampling import sample_next_token
from .request import GenerationRequest, RequestState


@dataclass(frozen=True)
class BatchStep:
    prefill: tuple[GenerationRequest, ...]
    decode: tuple[GenerationRequest, ...]


class NativeBatchRuntime:
    """Connect a Scheduler to a NativeQwen3 model.

    Decode executes before bounded, sequential prefill chunks. Shared complete
    prompt pages are immutable; an exact hit can sample cached boundary logits
    without running the model. A partial hit computes only its uncached suffix.
    """

    def __init__(self, model, scheduler, cache, *, prefix_cache=None, eos_token_ids=None,
                 graph_manager: DecodeGraphManager | None = None,
                 prefill_chunk_size: int | None = None, enable_prefix_cache=True,
                 sampling_backend="torch"):
        self.model = model
        self.scheduler = scheduler
        self.cache = cache
        self.prefix_cache = prefix_cache if prefix_cache is not None else PrefixCache(scheduler.block_manager)
        self.enable_prefix_cache = bool(enable_prefix_cache)
        if prefill_chunk_size is not None:
            if prefill_chunk_size <= 0:
                raise ValueError("prefill_chunk_size must be positive or None")
            scheduler.prefill_chunk_size = prefill_chunk_size
        if sampling_backend not in {"torch", "cuda"}:
            raise ValueError("sampling backend must be torch or cuda")
        self.sampling_backend = sampling_backend
        scheduler.prepare_request = self.admit_prefix
        scheduler.evict_cache = self.prefix_cache.evict_until_free
        self.eos_token_ids = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or [])
        self.graph_manager = graph_manager
        self.graph_padding_block_ids = None

    def prepare_waiting(self) -> None:
        for request in list(self.scheduler.waiting):
            limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
            if limit is not None and request.max_cache_tokens > limit:
                self.scheduler.abort(request.request_id)
                raise ValueError("prompt and output budget exceed model context limit")
            self.admit_prefix(request)

    def admit_prefix(self, request: GenerationRequest) -> int:
        request.prefix_cache_key = None
        request.prefix_block_ids = []
        request.prefix_len = 0
        request.prefix_next_logits = None
        if not self.enable_prefix_cache:
            return 0
        entry = self.prefix_cache.lookup(request.prompt_token_ids)
        if entry is not None and len(entry.token_key) == request.prompt_len and entry.next_logits is None:
            # Recomputing the final token would write a shared complete page.
            # Use a shorter valid prefix (or private recomputation) instead.
            entry = self.prefix_cache.lookup(request.prompt_token_ids[:-1])
        if entry is None:
            return 0
        request.prefix_cache_key = entry.token_key
        request.prefix_block_ids = list(entry.block_ids)
        request.prefix_len = len(entry.token_key)
        request.prefix_next_logits = entry.next_logits
        return request.prefix_len

    def _sample(self, logits, requests):
        params = [(r.temperature, r.top_k, r.top_p) for r in requests]
        if all(value == params[0] for value in params):
            return sample_next_token(logits, *params[0], backend=self.sampling_backend)
        return torch.cat([sample_next_token(logits[row:row + 1], *params[row], backend=self.sampling_backend)
                          for row in range(len(requests))])

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
        if self.graph_manager is None:
            logits = self.model.forward_decode_batch(tokens, self.cache, tables, positions)
        else:
            padding = self.graph_padding_block_ids
            if padding is None:
                raise RuntimeError("graph replay requires reserved graph_padding_block_ids")
            bucket = self.graph_manager.get(
                len(requests), tables.shape[1], padding_block_ids=padding
            )
            if not bucket.captured:
                bucket.capture()
            padded = torch.full(
                (len(requests), bucket.shape.max_blocks), -1,
                device=device, dtype=torch.long,
            )
            padded[:, :tables.shape[1]] = tables
            logits = bucket.replay(tokens, positions, padded, active_count=len(requests))
        chosen = self._sample(logits, requests).cpu().tolist()
        result = {}
        for row, request in enumerate(requests):
            token_id = int(chosen[row])
            finished = token_id in self.eos_token_ids
            self.scheduler.on_token(request, token_id, finished=finished)
            result[request.request_id] = token_id
        return result

    @torch.inference_mode()
    def prefill_step(self, requests: list[GenerationRequest]) -> dict[str, int]:
        """Execute reserved prompt ranges; emit only after the final chunk."""
        result: dict[str, int] = {}
        for request in requests:
            try:
                start, end = request.prefill_start, request.prefill_end
                if end > start:
                    prompt = torch.tensor(request.prompt_token_ids[start:end], device=self.cache.device, dtype=torch.long)
                    rows = self.model(prompt, self.cache, request.block_ids, position=start)
                    logits = rows[-1:]
                    boundary = end // self.scheduler.block_manager.block_size * self.scheduler.block_manager.block_size
                    if self.enable_prefix_cache and boundary > start:
                        self.prefix_cache.insert(
                            request.prompt_token_ids[:boundary], request.block_ids,
                            next_logits=rows[boundary - start - 1:boundary - start],
                        )
                else:
                    logits = request.prefix_next_logits
                    if logits is None:
                        raise RuntimeError("an exact prefix hit requires matching boundary logits")
                if end < request.prompt_len:
                    continue
                token_id = int(self._sample(logits, [request]).cpu().tolist()[0])
                self.scheduler.on_token(request, token_id, finished=token_id in self.eos_token_ids)
                result[request.request_id] = token_id
            except Exception:
                self.scheduler.abort(request.request_id)
                raise
        return result

    def step(self) -> dict[str, int]:
        """Execute decode first; then spend remaining token budget on prefill."""
        self.prepare_waiting()
        output = self.scheduler.schedule()
        active = [request for request in output.decode if not request.is_finished]
        try:
            generated = self.decode_step(active)
        except Exception:
            for request in active:
                self.scheduler.abort(request.request_id)
            # Ranges have already been reserved by schedule(). Do not leave
            # skipped prompt work marked as completed after a decode failure.
            for request in output.prefill:
                self.scheduler.abort(request.request_id)
            raise
        try:
            generated.update(self.prefill_step(list(output.prefill)))
        except Exception:
            for request in output.prefill:
                if not request.is_finished:
                    self.scheduler.abort(request.request_id)
            raise
        return generated

    def finish_request(self, request: GenerationRequest) -> None:
        if not request.is_finished:
            self.scheduler.abort(request.request_id)
            request.finish("stop")
