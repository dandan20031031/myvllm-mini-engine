"""Generation values shared by native execution and the optional reference runner."""

from dataclasses import dataclass


@dataclass
class IncrementalGenerationResult:
    text: str
    token_ids: list[int]
    finish_reason: str
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float
