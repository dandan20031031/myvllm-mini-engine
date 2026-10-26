"""Scheduler-driven native batch runtime for prefill/decode experiments."""

from __future__ import annotations

import torch

from ..cache import PrefixCache
from .batch import PrefillBatch
from .cuda_graph import DecodeGraphManager
from ..sampling import sample_next_token
from .request import GenerationRequest, RequestState


class NativeBatchRuntime:
    """Connect a Scheduler to a NativeQwen3 model.

    Decode executes before bounded, packed prefill chunks. Shared complete
    prompt pages are immutable; an exact hit can sample cached boundary logits
    without running the model. A partial hit computes only its uncached suffix.
    """

    def __init__(self, model, scheduler, cache, *, prefix_cache=None, eos_token_ids=None,
                 graph_manager: DecodeGraphManager | None = None,
                 prefill_chunk_size: int | None = None, enable_prefix_cache=True,
                 sampling_backend="torch", owns_graph_manager=False,
                 graph_padding_owner=None):
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
        self._previous_prepare_request = scheduler.prepare_request
        self._previous_evict_cache = scheduler.evict_cache
        scheduler.prepare_request = self.admit_prefix
        scheduler.evict_cache = self.prefix_cache.evict_until_free
        self.eos_token_ids = {eos_token_ids} if isinstance(eos_token_ids, int) else set(eos_token_ids or [])
        self.graph_manager = graph_manager
        self.graph_padding_block_ids = None
        self._owns_graph_manager = bool(owns_graph_manager)
        self._graph_padding_owner = graph_padding_owner
        self._owned_graph_buckets = {}
        self._replaced_graph_buckets = {}
        self._closed = False
        self._closing = False

    @property
    def closed(self) -> bool:
        return self._closed

    def _ensure_open(self) -> None:
        if self._closed or self._closing:
            raise RuntimeError("native batch runtime is closed")

    def close(self) -> None:
        """Release this session after its owning worker has stopped executing.

        Model weights and KV tensors belong to the runner. A borrowed graph
        manager and caller-provided scratch pages remain caller-owned. Borrowed
        managers cannot be used concurrently by another runtime. Buckets using
        this session's own scratch cannot survive page release.
        The assigned scheduler and prefix cache belong to this session and
        must not be shared concurrently; closing aborts and clears them.
        """
        if self._closed:
            return
        if self._closing:
            raise RuntimeError("native batch runtime is already closing")
        self._closing = True
        try:
            # Graph replay and metadata H2D can still be queued on another
            # CUDA stream. Do not recycle pages or drop buffers before it ends.
            if self.cache.device.type == "cuda":
                torch.cuda.synchronize(self.cache.device)
            requests = list(self.scheduler.waiting) + list(self.scheduler.running.values())
            for request in requests:
                self.scheduler.abort(request.request_id)
            self.prefix_cache.clear()
            manager = self.graph_manager
            if manager is not None:
                if self._owns_graph_manager:
                    for bucket in manager.buckets.values():
                        self._retire_graph_bucket(bucket)
                    manager.buckets.clear()
                else:
                    # Preserve borrowed manager buckets not tied to pages this
                    # runtime allocated. Replaying our retired buckets would
                    # otherwise write to pages that may now serve a new request.
                    for key, bucket in self._owned_graph_buckets.items():
                        if manager.buckets.get(key) is bucket:
                            self._retire_graph_bucket(bucket)
                            if key in self._replaced_graph_buckets:
                                manager.buckets[key] = self._replaced_graph_buckets[key]
                            else:
                                del manager.buckets[key]
            self._owned_graph_buckets.clear()
            self._replaced_graph_buckets.clear()
            self.graph_manager = None
            if self._graph_padding_owner is not None:
                self.scheduler.block_manager.release(self._graph_padding_owner)
                self._graph_padding_owner = None
            self.graph_padding_block_ids = None
            if self.scheduler.prepare_request == self.admit_prefix:
                self.scheduler.prepare_request = self._previous_prepare_request
            if self.scheduler.evict_cache == self.prefix_cache.evict_until_free:
                self.scheduler.evict_cache = self._previous_evict_cache
            self._closed = True
        finally:
            self._closing = False

    @staticmethod
    def _retire_graph_bucket(bucket) -> None:
        # A caller may still hold a bucket reference for diagnostics. Prevent
        # that stale reference from replaying writes after its pages are freed.
        for name in ("graph", "logits", "sampled_token_ids"):
            if hasattr(bucket, name):
                setattr(bucket, name, None)

    def prepare_waiting(self) -> None:
        self._ensure_open()
        for request in list(self.scheduler.waiting):
            limit = getattr(getattr(self.model, "config", None), "max_position_embeddings", None)
            if limit is not None and request.max_cache_tokens > limit:
                self.scheduler.abort(request.request_id)
                raise ValueError("prompt and output budget exceed model context limit")
            self.admit_prefix(request)

    def admit_prefix(self, request: GenerationRequest) -> int:
        self._ensure_open()
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

    def _get_graph_bucket(self, batch_size, max_blocks, padding):
        self._ensure_open()
        manager = self.graph_manager
        if self._owns_graph_manager or self._graph_padding_owner is None:
            return manager.get(batch_size, max_blocks, padding_block_ids=padding)
        key = manager.shape_for(batch_size, max_blocks)
        previous = manager.buckets.get(key)
        replace = previous is not None and previous.padding_block_ids is None
        if replace:
            # Do not attach runtime-owned pages or a new capture to a caller's
            # preexisting uninitialized bucket. Restore it when we close.
            del manager.buckets[key]
        try:
            bucket = manager.get(batch_size, max_blocks, padding_block_ids=padding)
        except BaseException:
            if replace:
                manager.buckets[key] = previous
            raise
        if previous is None or replace:
            self._owned_graph_buckets[key] = bucket
            if replace:
                self._replaced_graph_buckets[key] = previous
        return bucket

    @torch.inference_mode()
    def decode_step(self, requests: list[GenerationRequest]) -> dict[str, int]:
        self._ensure_open()
        if not requests:
            return {}
        device = self.cache.device
        token_ids = [request.output_token_ids[-1] for request in requests]
        offsets = [request.total_len - 1 for request in requests]
        width = max(len(request.block_ids) for request in requests)
        if self.graph_manager is None:
            tokens = torch.tensor(token_ids, device=device, dtype=torch.long)
            positions = torch.tensor(offsets, device=device, dtype=torch.long)
            tables = torch.full((len(requests), width), -1, device=device, dtype=torch.long)
            for row, request in enumerate(requests):
                tables[row, : len(request.block_ids)] = torch.tensor(request.block_ids, device=device)
            logits = self.model.forward_decode_batch(tokens, self.cache, tables, positions)
            chosen = self._sample(logits, requests).cpu().tolist()
        else:
            padding = self.graph_padding_block_ids
            if padding is None:
                raise RuntimeError("graph replay requires reserved graph_padding_block_ids")
            bucket = self._get_graph_bucket(len(requests), width, padding)
            if not bucket.captured:
                bucket.capture()
            greedy_in_graph = bucket.capture_sampling and all(request.temperature == 0 for request in requests)
            output = bucket.replay_metadata(token_ids, offsets, [request.block_ids for request in requests],
                                            sample=greedy_in_graph)
            chosen = (output if greedy_in_graph else self._sample(output, requests)).cpu().tolist()
        result = {}
        for row, request in enumerate(requests):
            token_id = int(chosen[row])
            finished = token_id in self.eos_token_ids
            self.scheduler.on_token(request, token_id, finished=finished)
            result[request.request_id] = token_id
        return result

    @torch.inference_mode()
    def prefill_step(self, requests: list[GenerationRequest]) -> dict[str, int]:
        """Pack scheduled prompt chunks; exact Prefix hits need no model call."""
        self._ensure_open()
        if not requests:
            return {}
        if not callable(getattr(self.model, "forward_prefill_batch", None)):
            # Small external/reference runners can retain the original model
            # protocol. Native Qwen3 always uses the packed implementation.
            return self._prefill_reference(requests)
        try:
            active = [request for request in requests if request.prefill_end > request.prefill_start]
            logits_by_request = {}
            if active:
                batch = PrefillBatch.from_sequences(
                    [request.prompt_token_ids[request.prefill_start:request.prefill_end] for request in active],
                    [request.block_ids for request in active],
                    [request.prefill_start for request in active],
                    block_size=self.scheduler.block_manager.block_size,
                    num_blocks=self.scheduler.block_manager.num_blocks, device=self.cache.device,
                )
                needed_rows, boundaries, last_rows = set(), {}, {}
                for index, request in enumerate(active):
                    start, end = request.prefill_start, request.prefill_end
                    offset = batch.query_offsets[index]
                    if end == request.prompt_len:
                        row = offset + end - start - 1
                        needed_rows.add(row)
                        last_rows[request.request_id] = row
                    boundary = end // batch.block_size * batch.block_size
                    if self.enable_prefix_cache and boundary > start:
                        row = offset + boundary - start - 1
                        needed_rows.add(row)
                        boundaries[request.request_id] = boundary, row
                selected = sorted(needed_rows)
                if len(active) == 1:
                    request = active[0]
                    # Preserve the public single-request hook/position contract.
                    rows = self.model(batch.input_ids, self.cache, position=request.prefill_start,
                                      prefill_batch=batch, logits_rows=selected)
                else:
                    rows = self.model.forward_prefill_batch(batch, self.cache, logits_rows=selected)
                lookup = {row: index for index, row in enumerate(selected)}
                for request in active:
                    if request.request_id in boundaries:
                        boundary, row = boundaries[request.request_id]
                        slot = lookup[row]
                        self.prefix_cache.insert(request.prompt_token_ids[:boundary], request.block_ids,
                                                 next_logits=rows[slot:slot + 1])
                    if request.request_id in last_rows:
                        slot = lookup[last_rows[request.request_id]]
                        logits_by_request[request.request_id] = rows[slot:slot + 1]
            ready, logits = [], []
            for request in requests:
                if request.prefill_end < request.prompt_len:
                    continue
                row = logits_by_request.get(request.request_id)
                if row is None:
                    row = request.prefix_next_logits
                if row is None:
                    raise RuntimeError("an exact prefix hit requires matching boundary logits")
                ready.append(request)
                logits.append(row)
            if not ready:
                return {}
            chosen = self._sample(torch.cat(logits, dim=0), ready).cpu().tolist()
            result = {}
            for request, token_id in zip(ready, chosen):
                token_id = int(token_id)
                self.scheduler.on_token(request, token_id, finished=token_id in self.eos_token_ids)
                result[request.request_id] = token_id
            return result
        except Exception:
            # A packed execution failure cannot safely be attributed to a row.
            for request in requests:
                if not request.is_finished:
                    self.scheduler.abort(request.request_id)
            raise

    def _prefill_reference(self, requests):
        """Compatibility protocol for non-native reference/test runners."""
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
        self._ensure_open()
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
