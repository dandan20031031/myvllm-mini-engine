from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch

from ..cache.block_manager import BlockManager
from ..cache.kv_cache import KVCachePool
from ..config import EngineConfig
from ..model.qwen3 import Qwen3ModelRunner
from ..quantization.backends import create_backend
from .request import GenerationRequest
from .scheduler import Scheduler


@dataclass
class GenerationResult:
    request_id: str
    text: str
    token_ids: list[int]
    finish_reason: str


class LLMEngine:
    """Public engine facade.

    The scheduler and cache lifecycle are functional now.  Model execution is
    intentionally kept behind a runner so custom Qwen3/CUDA execution can be
    introduced without changing this API.
    """

    def __init__(self, config: EngineConfig | None = None, model_runner=None):
        self.config = config or EngineConfig()
        self.block_manager = BlockManager(self.config.num_gpu_blocks, self.config.block_size)
        # The model runner can replace this with architecture-specific values
        # once a Qwen3 config is loaded.  These defaults are useful for unit
        # tests and keep cache allocation explicit instead of hidden.
        self.kv_cache: KVCachePool | None = None
        self.scheduler = Scheduler(self.block_manager, self.config.max_num_batched_tokens)
        self._ids = itertools.count()
        self.quant_backend = create_backend(self.config.quantization)
        self.model_runner = model_runner

    def allocate_kv_cache(
        self,
        *,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype | None = None,
    ) -> KVCachePool:
        """Allocate the physical cache used by a model runner."""
        self.kv_cache = KVCachePool(
            num_layers=num_layers,
            num_blocks=self.config.num_gpu_blocks,
            block_size=self.config.block_size,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            dtype=dtype or torch.bfloat16,
            device=self.config.device,
        )
        return self.kv_cache

    def add_request(self, prompt_token_ids: list[int], **kwargs) -> GenerationRequest:
        request = GenerationRequest(
            request_id=f"req-{next(self._ids)}",
            prompt_token_ids=list(prompt_token_ids),
            max_new_tokens=kwargs.get("max_new_tokens", self.config.max_new_tokens),
            temperature=kwargs.get("temperature", 0.0),
            top_p=kwargs.get("top_p", 1.0),
            top_k=kwargs.get("top_k", 0),
        )
        self.scheduler.add_request(request)
        return request

    def schedule(self):
        return self.scheduler.schedule()

    def step_reference(self):
        """Run one scheduling step without pretending model execution is done."""
        if self.model_runner is None:
            raise RuntimeError("no model_runner configured; use schedule() for cache/scheduler tests")
        return self.scheduler.schedule()

    def generate_reference(self, prompt: str, max_new_tokens: int | None = None, **kwargs) -> str:
        """Run the HuggingFace reference path when a Qwen3 runner is attached."""
        if self.model_runner is None:
            raise RuntimeError("no model_runner configured")
        return self.model_runner.generate(
            prompt,
            max_new_tokens=self.config.max_new_tokens if max_new_tokens is None else max_new_tokens,
            **kwargs,
        )

    def generate_incremental(self, prompt: str, max_new_tokens: int | None = None, **kwargs):
        """Run the reference prefill/decode loop and return timing metrics."""
        if self.model_runner is None or not hasattr(self.model_runner, "generate_incremental"):
            raise RuntimeError("an incremental-capable model_runner is required")
        return self.model_runner.generate_incremental(
            prompt,
            max_new_tokens=self.config.max_new_tokens if max_new_tokens is None else max_new_tokens,
            **kwargs,
        )
