"""Windows-native Qwen3 inference with explicit startup configuration."""

from .runtime import configure_local_cache

configure_local_cache()  # Before importing torch/model code or initializing CUDA.

from .config import QuantConfig, RunnerConfig

__all__ = ["QuantConfig", "RunnerConfig"]
