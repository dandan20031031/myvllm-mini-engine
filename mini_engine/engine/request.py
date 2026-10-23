from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class RequestState(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    FINISHED = "finished"
    ABORTED = "aborted"


@dataclass
class GenerationRequest:
    request_id: str
    prompt_token_ids: list[int]
    max_new_tokens: int = 128
    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    state: RequestState = RequestState.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    block_ids: list[int] = field(default_factory=list)
    generated_tokens: int = 0
    finished_reason: Optional[str] = None
    prefix_len: int = 0
    prefix_cache_key: Optional[tuple[int, ...]] = None
    prefix_block_ids: list[int] = field(default_factory=list)
    # Prompt ranges scheduled on the current tick. Cursor advances when the
    # range is reserved; a failed executor aborts the request and releases it.
    prefill_cursor: int = 0
    prefill_start: int = 0
    prefill_end: int = 0
    prefix_next_logits: object | None = field(default=None, repr=False)

    @property
    def prompt_len(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def total_len(self) -> int:
        return self.prompt_len + self.generated_tokens

    @property
    def max_cache_tokens(self) -> int:
        # The last emitted token never needs a forward/KV write.
        return self.prompt_len + self.max_new_tokens - 1

    @property
    def is_finished(self) -> bool:
        return self.state in (RequestState.FINISHED, RequestState.ABORTED)

    def append_token(self, token_id: int) -> None:
        self.output_token_ids.append(int(token_id))
        self.generated_tokens += 1

    def finish(self, reason: str = "stop") -> None:
        self.state = RequestState.FINISHED
        self.finished_reason = reason
