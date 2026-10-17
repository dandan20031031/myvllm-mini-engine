from collections import deque
from dataclasses import dataclass

from ..cache.block_manager import BlockManager
from .request import GenerationRequest, RequestState


@dataclass(frozen=True)
class SchedulerOutput:
    prefill: tuple[GenerationRequest, ...]
    decode: tuple[GenerationRequest, ...]


class Scheduler:
    """Small continuous-batching scheduler.

    Each step reserves enough blocks for the prompt on admission, then grows
    the request's block table as generated tokens arrive.  Decode requests are
    returned first so a long prompt cannot starve active streams.
    """

    def __init__(self, block_manager: BlockManager, max_num_batched_tokens: int = 4096):
        self.block_manager = block_manager
        self.max_num_batched_tokens = max_num_batched_tokens
        self.waiting: deque[GenerationRequest] = deque()
        self.running: dict[str, GenerationRequest] = {}

    def add_request(self, request: GenerationRequest) -> None:
        if request.request_id in self.running or any(r.request_id == request.request_id for r in self.waiting):
            raise ValueError(f"duplicate request id: {request.request_id}")
        self.waiting.append(request)

    def abort(self, request_id: str) -> None:
        request = self.running.pop(request_id, None)
        if request is not None:
            request.state = RequestState.ABORTED
            self.block_manager.release(request_id)
            return
        for request in self.waiting:
            if request.request_id == request_id:
                request.state = RequestState.ABORTED
                return

    def schedule(self) -> SchedulerOutput:
        prefill: list[GenerationRequest] = []
        budget = self.max_num_batched_tokens
        for request in list(self.waiting):
            if request.state is RequestState.ABORTED:
                self.waiting.remove(request)
                continue
            if request.prompt_len > budget:
                break
            if not self.block_manager.can_allocate(request.prompt_len):
                break
            self.waiting.remove(request)
            request.state = RequestState.RUNNING
            request.block_ids = self.block_manager.allocate(request.request_id, request.prompt_len)
            self.running[request.request_id] = request
            prefill.append(request)
            budget -= request.prompt_len

        decode: list[GenerationRequest] = []
        for request in self.running.values():
            if request not in prefill and not request.is_finished:
                decode.append(request)
        return SchedulerOutput(tuple(prefill), tuple(decode))

    def on_token(self, request: GenerationRequest, token_id: int, finished: bool = False) -> None:
        request.append_token(token_id)
        request.block_ids = self.block_manager.grow(request.request_id, request.total_len)
        if finished or request.generated_tokens >= request.max_new_tokens:
            request.finish("length" if not finished else "stop")
            self.running.pop(request.request_id, None)
            self.block_manager.release(request.request_id)
