"""Small, backend-agnostic latency benchmark utilities.

The runner accepts a callable so it can benchmark Transformers, vLLM and the
native engine through adapters with identical timing and output formats.
"""

from __future__ import annotations

import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Sequence

import torch


@dataclass(frozen=True)
class BenchmarkConfig:
    name: str
    warmup: int = 1
    repetitions: int = 3
    output_tokens: int = 32


@dataclass
class BenchmarkResult:
    name: str
    repetitions: int
    latencies_ms: list[float]
    mean_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    tokens_per_second: float
    peak_memory_mb: float | None
    output_tokens: int

    def to_dict(self) -> dict:
        return asdict(self)


def _sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize(torch.device(device))


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * percentile
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    fraction = position - low
    return ordered[low] + (ordered[high] - ordered[low]) * fraction


def benchmark_callable(
    fn: Callable[[], object],
    config: BenchmarkConfig,
    *,
    device: str = "cpu",
) -> BenchmarkResult:
    """Benchmark a callable after warmup and return JSON-friendly metrics."""
    if config.warmup < 0 or config.repetitions <= 0 or config.output_tokens <= 0:
        raise ValueError("warmup >= 0, repetitions > 0 and output_tokens > 0 are required")
    for _ in range(config.warmup):
        fn()
    _sync(device)
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(torch.device(device))
    latencies: list[float] = []
    for _ in range(config.repetitions):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        latencies.append((time.perf_counter() - start) * 1000.0)
    mean_ms = statistics.fmean(latencies)
    peak = None
    if device.startswith("cuda") and torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated(torch.device(device)) / (1024 * 1024)
    return BenchmarkResult(
        name=config.name,
        repetitions=config.repetitions,
        latencies_ms=latencies,
        mean_latency_ms=mean_ms,
        p50_latency_ms=_percentile(latencies, 0.50),
        p95_latency_ms=_percentile(latencies, 0.95),
        tokens_per_second=config.output_tokens / (mean_ms / 1000.0),
        peak_memory_mb=peak,
        output_tokens=config.output_tokens,
    )


def save_results(results: Sequence[BenchmarkResult], path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps([item.to_dict() for item in results], indent=2), encoding="utf-8")
