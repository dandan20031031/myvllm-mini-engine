"""Single-request native generation; cache ownership is released on all exits."""

import time

import torch

from ..cache import BlockManager
from ..sampling import sample_next_token
from .native_qwen3 import NativeQwen3
from .qwen3 import IncrementalGenerationResult


class NativeQwen3Runner:
    def __init__(self, model, tokenizer, *, num_blocks=256, block_size=16, enable_thinking=False, eos_token_ids=None,
                 sampling_backend="torch"):
        self.model = model.eval()
        self.tokenizer = tokenizer
        self.enable_thinking = enable_thinking
        self.block_manager = BlockManager(num_blocks, block_size)
        self.cache = model.create_cache(num_blocks, block_size)
        self.eos_token_ids = eos_token_ids
        self._busy = False
        self.last_backend = model.decode_backend
        if sampling_backend not in {"torch", "cuda"}:
            raise ValueError("sampling backend must be torch or cuda")
        self.sampling_backend = sampling_backend

    @classmethod
    def from_pretrained(cls, path, *, device="cuda", dtype=torch.bfloat16, prefill_backend="auto", decode_backend="torch", elementwise_backend="torch", data_backend="torch", linear_backend="torch",
                        num_blocks=256, block_size=16, enable_thinking=False, local_files_only=False,
                        quantization=None, fuse_projections=False, quant_backend="reference", sampling_backend="torch"):
        from ..runtime import configure_local_cache
        configure_local_cache()
        if sampling_backend not in {"torch", "cuda"}:
            raise ValueError("sampling backend must be torch or cuda")
        if sampling_backend == "cuda":
            if not str(device).startswith("cuda"):
                raise ValueError("CUDA sampling requires CUDA device")
            from ..kernels.cuda_extension import load_cuda_extension
            load_cuda_extension()
        from ..kernels.dandan_backend import DANDAN_LINEAR_BACKENDS, LINEAR_BACKENDS, validate_dandan_runtime
        if linear_backend not in LINEAR_BACKENDS:
            raise ValueError("unknown linear backend")
        if quant_backend not in {"reference", "dandan"}:
            raise ValueError("quant backend must be reference or dandan")
        if quant_backend == "dandan":
            from ..quantization.dandan import validate_quant_config
            validate_quant_config(quantization)
            if dtype != torch.float16:
                raise ValueError("dandan quant backend requires FP16 weights/activations")
        if quant_backend == "reference" and quantization is not None and fuse_projections:
            raise ValueError("reference quantization and projection fusion cannot be combined yet")
        if linear_backend in DANDAN_LINEAR_BACKENDS or quant_backend == "dandan" or prefill_backend == "dandan":
            validate_dandan_runtime(device, dtype)
        if prefill_backend == "dandan" or quant_backend == "dandan":
            from ..kernels.dandan_backend import require_dandan_capability
            if prefill_backend == "dandan":
                require_dandan_capability("prefill")
            if quant_backend == "dandan":
                require_dandan_capability("int4")
        if quantization is not None:
            from ..quantization import create_backend
            create_backend(quantization)  # Reject unavailable formats before downloading weights.
        if decode_backend == "cuda":
            if not str(device).startswith("cuda") or dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("CUDA decode requires CUDA device and FP16/BF16 weights")
            from ..kernels.cuda_extension import load_cuda_extension
            load_cuda_extension()  # compile/load before any token timing
        if elementwise_backend == "cuda":
            if not str(device).startswith("cuda") or dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("CUDA elementwise kernels require CUDA device and FP16/BF16 weights")
            from ..kernels.cuda_extension import load_cuda_extension
            load_cuda_extension()
        if data_backend == "cuda":
            if not str(device).startswith("cuda") or dtype not in (torch.float16, torch.bfloat16):
                raise ValueError("CUDA data kernels require CUDA device and FP16/BF16 weights")
            from ..kernels.cuda_extension import load_cuda_extension
            load_cuda_extension()
        if linear_backend == "cuda" and not str(device).startswith("cuda"):
            raise ValueError("CUDA linear backend requires a CUDA device")
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if prefill_backend == "dandan":
            from transformers import AutoConfig
            config = AutoConfig.from_pretrained(path, local_files_only=local_files_only)
            head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
            if head_dim != 128:
                raise ValueError("dandan prefill requires head_dim=128")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=local_files_only)
        reference = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=dtype, local_files_only=local_files_only,
        ).to(device).eval()
        native = NativeQwen3.from_hf_model(reference, prefill_backend, decode_backend, elementwise_backend, data_backend, linear_backend)
        if prefill_backend == "dandan" and native.config.head_dim != 128:
            raise ValueError("dandan prefill requires head_dim=128")
        if fuse_projections:
            native.enable_projection_fusion()
        stop_ids = reference.generation_config.eos_token_id
        del reference
        if quantization is not None:
            native.quantize(quantization, backend=quant_backend)
        return cls(native, tokenizer, num_blocks=num_blocks, block_size=block_size,
                   enable_thinking=enable_thinking, eos_token_ids=stop_ids, sampling_backend=sampling_backend)

    def _sync(self):
        if self.cache.device.type == "cuda":
            torch.cuda.synchronize(self.cache.device)

    @torch.inference_mode()
    def generate_incremental(self, prompt=None, max_new_tokens=128, *, temperature=0.0, top_k=0, top_p=1.0, messages=None):
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        if self._busy:
            raise RuntimeError("native runner currently supports one active request")
        if messages is None:
            if prompt is None:
                raise ValueError("prompt or messages is required")
            messages = [{"role": "user", "content": prompt}]
        ids = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            enable_thinking=self.enable_thinking, return_tensors="pt",
        )[0].to(self.cache.device)
        required = ids.numel() + max_new_tokens - 1
        if required > self.model.config.max_position_embeddings:
            raise ValueError("prompt and output budget exceed context limit")
        if not self.block_manager.can_allocate(required):
            raise MemoryError("prompt and output budget exceed KV cache capacity")
        eos = self.eos_token_ids
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos = {eos} if isinstance(eos, int) else set(eos or [])
        self._busy = True
        generated, first = [], None
        try:
            # Reserve only the prompt now. The complete budget was checked above,
            # so growing a single active request cannot deadlock midway.
            blocks = self.block_manager.allocate("single", ids.numel())
            self._sync()
            start = time.perf_counter()
            logits = self.model(ids, self.cache, blocks)[-1:]
            position = ids.numel()
            reason = "length"
            for index in range(max_new_tokens):
                token = sample_next_token(logits, temperature, top_k, top_p, backend=self.sampling_backend)
                token_id = int(token.item())
                self._sync()
                emitted = time.perf_counter()
                if first is None:
                    first = emitted
                generated.append(token_id)
                if token_id in eos:
                    reason = "stop"
                    break
                if index + 1 == max_new_tokens:
                    break
                blocks = self.block_manager.grow("single", position + 1)
                logits = self.model(token, self.cache, blocks, position=position)
                position += 1
            end = emitted
            return IncrementalGenerationResult(
                text=self.tokenizer.decode(generated, skip_special_tokens=True),
                token_ids=generated, finish_reason=reason,
                ttft_ms=(first - start) * 1000,
                tpot_ms=(end - first) * 1000 / (len(generated) - 1) if len(generated) > 1 else 0.0,
                total_latency_ms=(end - start) * 1000,
            )
        finally:
            self.block_manager.release("single")
            self._busy = False

    def generate(self, prompt, max_new_tokens=128, **kwargs):
        return self.generate_incremental(prompt, max_new_tokens, **kwargs).text

    def chat(self, messages, max_new_tokens=128, **kwargs):
        """Generate one assistant turn from a Qwen chat-template message list."""
        result = self.generate_incremental(None, max_new_tokens, messages=list(messages), **kwargs)
        return result.text

    def generate_speculative(self, prompt=None, max_new_tokens=128, **kwargs):
        """Greedy prompt-lookup drafting with target verification and KV rollback."""
        from ..speculative import generate_speculative
        return generate_speculative(self, prompt, max_new_tokens, **kwargs)

    def create_batch_runtime(self, *, max_prefix_entries=128, graph_manager=None,
                             graph_padding_block_ids=None, graph_padding_blocks=0,
                             use_cuda_graph=False, graph_batch_sizes=(1, 2, 4, 8),
                             graph_block_buckets=(16, 32, 64), prefill_chunk_size=None,
                             max_num_batched_tokens=4096, enable_prefix_cache=True):
        """Create the scheduler-driven runtime over this runner's model/cache."""
        from ..cache import PrefixCache
        from ..engine import NativeBatchRuntime
        from ..engine.scheduler import Scheduler

        scheduler = Scheduler(self.block_manager, max_num_batched_tokens=max_num_batched_tokens,
                              prefill_chunk_size=prefill_chunk_size)
        prefix = PrefixCache(self.block_manager, max_entries=max_prefix_entries)
        if use_cuda_graph and graph_manager is None:
            from ..engine import DecodeGraphManager
            graph_manager = DecodeGraphManager(
                self.model, self.cache,
                batch_sizes=graph_batch_sizes,
                block_buckets=graph_block_buckets,
            )
        if graph_manager is not None and graph_padding_block_ids is None:
            if graph_padding_blocks <= 0:
                graph_padding_blocks = max(max(graph_manager.block_buckets), max(getattr(graph_manager, "batch_sizes", (1,))))
            graph_padding_block_ids = self.block_manager.allocate(
                # BlockManager.allocate receives a token count, while the
                # graph manager expresses scratch capacity in physical pages.
                "__graph_padding__", graph_padding_blocks * self.block_manager.block_size
            )
        runtime = NativeBatchRuntime(
            self.model, scheduler, self.cache,
            prefix_cache=prefix, eos_token_ids=self.eos_token_ids,
            graph_manager=graph_manager,
            prefill_chunk_size=prefill_chunk_size, enable_prefix_cache=enable_prefix_cache,
            sampling_backend=self.sampling_backend,
        )
        runtime.graph_padding_block_ids = graph_padding_block_ids
        return runtime
