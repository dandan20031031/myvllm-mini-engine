"""Decode-priority scheduling with bounded prompt chunks and KV reservations."""

from collections import deque
from dataclasses import dataclass

from ..cache.block_manager import BlockManager
from .request import GenerationRequest, RequestState


@dataclass(frozen=True)
class SchedulerOutput:
    prefill: tuple[GenerationRequest, ...]
    decode: tuple[GenerationRequest, ...]


class Scheduler:
    """Reserve worst-case growth virtually; allocate physical pages on demand.

    A reservation prevents newly admitted prompts consuming pages needed by
    existing streams. Cached pages can be evicted by the runtime callback;
    request references remain untouched. Execute returned work before the
    next tick, and abort requests whose execution fails.
    """

    def __init__(self, block_manager: BlockManager, max_num_batched_tokens: int = 4096,
                 prefill_chunk_size: int | None = None):
        if max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if prefill_chunk_size is not None and prefill_chunk_size <= 0:
            raise ValueError("prefill_chunk_size must be positive or None")
        self.block_manager = block_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.prefill_chunk_size = prefill_chunk_size
        self.waiting: deque[GenerationRequest] = deque()
        self.running: dict[str, GenerationRequest] = {}
        self.prepare_request = None
        self.evict_cache = None

    def add_request(self, request: GenerationRequest) -> None:
        if request.request_id in self.running or any(r.request_id == request.request_id for r in self.waiting):
            raise ValueError(f"duplicate request id: {request.request_id}")
        if request.state is not RequestState.WAITING or not request.prompt_token_ids or request.max_new_tokens <= 0:
            raise ValueError("new requests require a nonempty prompt and positive output budget")
        if self.block_manager.blocks_required(request.max_cache_tokens) > self.block_manager.num_blocks:
            raise MemoryError("request prompt and output budget exceed physical KV capacity")
        self.waiting.append(request)

    def abort(self, request_id: str) -> None:
        request = self.running.pop(request_id, None)
        if request is None:
            request = next((r for r in self.waiting if r.request_id == request_id), None)
            if request is not None:
                self.waiting.remove(request)
        if request is not None:
            request.state = RequestState.ABORTED
            request.finished_reason = "aborted"
            self.block_manager.release(request_id)

    def _ensure_free(self, count: int, protected_keys=()) -> bool:
        if self.block_manager.free_blocks >= count:
            return True
        return bool(self.evict_cache is not None and self.evict_cache(count, protected_keys=protected_keys))

    def _pending_growth(self) -> int:
        return sum(max(0, self.block_manager.blocks_required(r.max_cache_tokens) - len(r.block_ids))
                   for r in self.running.values() if not r.is_finished)

    def _schedule_prefill(self, request: GenerationRequest, budget: int) -> int:
        remaining = request.prompt_len - request.prefill_cursor
        if remaining and budget <= 0:
            return 0
        size = min(remaining, budget, self.prefill_chunk_size or self.max_num_batched_tokens)
        request.prefill_start = request.prefill_cursor
        request.prefill_end = request.prefill_cursor + size
        request.prefill_cursor = request.prefill_end
        request.block_ids = self.block_manager.grow(request.request_id, request.prefill_end)
        return size

    def schedule(self) -> SchedulerOutput:
        prefill, decode = [], []
        budget = self.max_num_batched_tokens
        ready = [r for r in self.running.values() if not r.is_finished and r.prefill_cursor == r.prompt_len]
        for request in ready[:budget]:
            needed = self.block_manager.blocks_required(max(request.prompt_len, request.total_len)) - len(request.block_ids)
            if not self._ensure_free(max(needed, 0)):
                raise MemoryError("reserved decode KV pages are unavailable")
            request.block_ids = self.block_manager.grow(request.request_id, max(request.prompt_len, request.total_len))
            decode.append(request)
            budget -= 1
            self.running.pop(request.request_id)
            self.running[request.request_id] = request

        for request in list(self.running.values()):
            if request in decode or request.is_finished or request.prefill_cursor >= request.prompt_len:
                continue
            if budget <= 0:
                break
            used = self._schedule_prefill(request, budget)
            prefill.append(request)
            budget -= used

        for request in list(self.waiting):
            if request.is_finished:
                self.waiting.remove(request)
                continue
            # Refresh immediately before admission: evicting another entry
            # can invalidate earlier metadata of a waiting request.
            if self.prepare_request is not None:
                self.prepare_request(request)
            remaining = request.prompt_len - request.prefix_len
            if remaining and budget <= 0:
                continue
            full_blocks = self.block_manager.blocks_required(request.max_cache_tokens)
            admission_need = self._pending_growth() + full_blocks - len(request.prefix_block_ids)
            protected = (request.prefix_cache_key,) if request.prefix_cache_key is not None else ()
            if not self._ensure_free(admission_need, protected):
                if not self.running:
                    self.abort(request.request_id)
                    raise MemoryError("request output budget exceeds KV capacity after reserved/scratch pages")
                continue
            request.prefill_cursor = request.prefix_len
            request.block_ids = self.block_manager.allocate_with_prefix(
                request.request_id, request.prefix_len, request.prefix_block_ids,
            )
            self.waiting.remove(request)
            request.state = RequestState.RUNNING
            self.running[request.request_id] = request
            used = self._schedule_prefill(request, budget)
            prefill.append(request)
            budget -= used
        return SchedulerOutput(tuple(prefill), tuple(decode))

    def on_token(self, request: GenerationRequest, token_id: int, finished: bool = False) -> None:
        request.append_token(token_id)
        if finished or request.generated_tokens >= request.max_new_tokens:
            request.finish("stop" if finished else "length")
            self.running.pop(request.request_id, None)
            self.block_manager.release(request.request_id)
            return
        needed = self.block_manager.blocks_required(request.total_len) - len(request.block_ids)
        if not self._ensure_free(max(0, needed)):
            self.abort(request.request_id)
            raise MemoryError("reserved generation KV pages are unavailable")
        request.block_ids = self.block_manager.grow(request.request_id, request.total_len)
