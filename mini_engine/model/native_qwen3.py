"""Dense Qwen3 with packed prefill and project-owned paged KV attention.

HuggingFace is used only for loading configuration, weights and tokenizer.
No Transformers model forward or past_key_values is used here.
"""

from dataclasses import dataclass

import torch
from torch import nn

from ..cache import KVCachePool
from ..kernels.data import apply_rope as apply_rope_backend, embedding_lookup, write_kv_cache_batch, write_kv_cache_slots
from ..kernels.elementwise import residual_add, residual_rms_norm, rms_norm, silu_and_mul
from ..kernels.linear import FusedLinear, make_linear
from ..kernels.dandan_backend import DANDAN_LINEAR_BACKENDS, LINEAR_BACKENDS
from ..kernels.paged import paged_attention
from ..kernels.prefill import paged_prefill_attention, prefill_attention


@dataclass(frozen=True)
class Qwen3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 32768
    attention_bias: bool = False
    tie_word_embeddings: bool = True

    def __post_init__(self):
        for name in ("vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
                     "num_attention_heads", "num_key_value_heads", "head_dim", "max_position_embeddings"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError("Qwen3 requires divisible GQA heads and an even head_dim")

    @classmethod
    def from_hf(cls, config):
        if config.model_type != "qwen3":
            raise ValueError("only dense Qwen3 is currently supported")
        if config.hidden_act != "silu":
            raise ValueError("only SiLU MLP is currently supported")
        if getattr(config, "rope_scaling", None):
            raise ValueError("scaled RoPE is not implemented")
        if getattr(config, "use_sliding_window", False) or "sliding_attention" in getattr(config, "layer_types", []):
            raise ValueError("sliding-window attention is not implemented")
        values = {name: getattr(config, name) for name in cls.__dataclass_fields__ if hasattr(config, name)}
        values["head_dim"] = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
        return cls(**values)


class RMSNorm(nn.Module):
    def __init__(self, size, eps, backend="torch"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps
        self.backend = backend

    def forward(self, x):
        return rms_norm(x, self.weight, self.eps, backend=self.backend)


class Attention(nn.Module):
    def __init__(self, config, elementwise_backend="torch", data_backend="torch", linear_backend="torch"):
        super().__init__()
        self.config = config
        self.data_backend = data_backend
        self.linear_backend = linear_backend
        self.q_proj = make_linear(config.hidden_size, config.num_attention_heads * config.head_dim, bias=config.attention_bias, backend=linear_backend)
        self.k_proj = make_linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias, backend=linear_backend)
        self.v_proj = make_linear(config.hidden_size, config.num_key_value_heads * config.head_dim, bias=config.attention_bias, backend=linear_backend)
        self.o_proj = make_linear(config.num_attention_heads * config.head_dim, config.hidden_size, bias=config.attention_bias, backend=linear_backend)
        self.q_norm = RMSNorm(config.head_dim, config.rms_norm_eps, elementwise_backend)
        self.k_norm = RMSNorm(config.head_dim, config.rms_norm_eps, elementwise_backend)
        self.qkv_proj = None

    @property
    def projection_fused(self):
        return self.qkv_proj is not None

    def fuse_qkv(self):
        """Replace Q/K/V projections with one concatenated matmul."""
        if self.qkv_proj is not None:
            return False
        self.qkv_proj = FusedLinear.from_linears(
            (self.q_proj, self.k_proj, self.v_proj), backend=self.linear_backend
        )
        del self.q_proj
        del self.k_proj
        del self.v_proj
        return True

    def _project_qkv(self, x):
        cfg = self.config
        if self.qkv_proj is None:
            q = self.q_proj(x).view(x.shape[0], cfg.num_attention_heads, cfg.head_dim)
            k = self.k_proj(x).view(x.shape[0], cfg.num_key_value_heads, cfg.head_dim)
            v = self.v_proj(x).view(x.shape[0], cfg.num_key_value_heads, cfg.head_dim)
            return q, k, v
        q_size = cfg.num_attention_heads * cfg.head_dim
        k_size = cfg.num_key_value_heads * cfg.head_dim
        qkv = self.qkv_proj(x)
        q_raw, k_raw, v_raw = qkv.split((q_size, k_size, k_size), dim=-1)
        return (
            q_raw.contiguous().view(x.shape[0], cfg.num_attention_heads, cfg.head_dim),
            k_raw.contiguous().view(x.shape[0], cfg.num_key_value_heads, cfg.head_dim),
            v_raw.contiguous().view(x.shape[0], cfg.num_key_value_heads, cfg.head_dim),
        )

    def forward(self, x, cos, sin, cache, block_ids, position, layer, backend, decode_backend, *, prefill_batch=None):
        if prefill_batch is not None:
            return self._forward_prefill_batch(x, cos, sin, cache, prefill_batch, layer, backend)
        cfg = self.config
        length = x.shape[0]
        q, k, v = self._project_qkv(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        rope_cos = cos.squeeze(1) if self.data_backend == "cuda" else cos
        rope_sin = sin.squeeze(1) if self.data_backend == "cuda" else sin
        q, k = apply_rope_backend(q, k, rope_cos, rope_sin, backend=self.data_backend, validate=False)
        if self.data_backend == "cuda":
            cache.write_cuda(layer, block_ids, position, k, v, validate=False)
        else:
            cache.write(layer, block_ids, position, k, v)
        if position == 0 or length > 1:
            if position > 0:
                # Only multi-token suffix prefill/verification gathers KV.
                # One-token decode continues to read physical pages directly.
                k, v = cache.gather(layer, block_ids, position + length)
            output = prefill_attention(
                q.transpose(0, 1).unsqueeze(0), k.transpose(0, 1).unsqueeze(0),
                v.transpose(0, 1).unsqueeze(0), backend, query_start=position,
            ).squeeze(0).transpose(0, 1)
        else:
            table = torch.as_tensor([block_ids], device=x.device, dtype=torch.long)
            lengths = torch.tensor([position + 1], device=x.device, dtype=torch.long)
            # Cache.write already validates all physical block ids; position
            # and logical capacity were checked before accessing the cache.
            output = paged_attention(q, cache.key[layer], cache.value[layer], table, lengths,
                                     cache.shape.block_size, backend=decode_backend, validate=False)
        return self.o_proj(output.reshape(length, -1))

    def _forward_prefill_batch(self, x, cos, sin, cache, batch, layer, backend):
        q, k, v = self._project_qkv(x)
        q, k = self.q_norm(q), self.k_norm(k)
        rope_cos = cos if self.data_backend == "cuda" else cos.unsqueeze(1)
        rope_sin = sin if self.data_backend == "cuda" else sin.unsqueeze(1)
        q, k = apply_rope_backend(q, k, rope_cos, rope_sin, backend=self.data_backend, validate=False)
        write_kv_cache_slots(cache.key[layer], cache.value[layer], batch.slot_mapping, k, v,
                             backend=self.data_backend, validate=False)
        if backend == "cuda":
            output = paged_prefill_attention(q, cache.key[layer], cache.value[layer],
                batch.cu_seqlens_q, batch.query_starts, batch.block_tables, batch.context_lens,
                cache.shape.block_size, backend="cuda", validate=False)
        else:
            # Preserve strict flash/dandan selection. Their current interfaces
            # are dense, so only projection/normalization/MLP are packed here.
            chunks = []
            for request, (start, length) in enumerate(zip(batch.starts, batch.query_lengths)):
                left, right = batch.query_offsets[request:request + 2]
                if start:
                    keys, values = cache.gather(layer, batch.block_ids[request], start + length)
                else:
                    keys, values = k[left:right], v[left:right]
                attention = prefill_attention(q[left:right].transpose(0, 1).unsqueeze(0),
                    keys.transpose(0, 1).unsqueeze(0), values.transpose(0, 1).unsqueeze(0),
                    backend, query_start=start)
                chunks.append(attention.squeeze(0).transpose(0, 1))
            output = torch.cat(chunks, dim=0)
        return self.o_proj(output.reshape(x.shape[0], -1))

    def forward_decode_batch(self, x, cos, sin, cache, block_tables, positions, layer, decode_backend):
        cfg = self.config
        batch = x.shape[0]
        q, k, v = self._project_qkv(x)
        q = self.q_norm(q)
        k = self.k_norm(k)
        rope_cos = cos if self.data_backend == "cuda" else cos.unsqueeze(1)
        rope_sin = sin if self.data_backend == "cuda" else sin.unsqueeze(1)
        q, k = apply_rope_backend(q, k, rope_cos, rope_sin, backend=self.data_backend, validate=False)
        write_kv_cache_batch(
            cache.key[layer], cache.value[layer], block_tables, positions,
            k, v, cache.shape.block_size, backend=self.data_backend, validate=False,
        )
        lengths = positions + 1
        output = paged_attention(
            q, cache.key[layer], cache.value[layer], block_tables, lengths,
            cache.shape.block_size, backend=decode_backend, validate=False,
        )
        return self.o_proj(output.reshape(batch, -1))


class MLP(nn.Module):
    def __init__(self, config, elementwise_backend="torch", linear_backend="torch"):
        super().__init__()
        self.gate_proj = make_linear(config.hidden_size, config.intermediate_size, bias=False, backend=linear_backend)
        self.up_proj = make_linear(config.hidden_size, config.intermediate_size, bias=False, backend=linear_backend)
        self.down_proj = make_linear(config.intermediate_size, config.hidden_size, bias=False, backend=linear_backend)
        self.elementwise_backend = elementwise_backend
        self.linear_backend = linear_backend
        self.gate_up_proj = None

    @property
    def projection_fused(self):
        return self.gate_up_proj is not None

    def fuse_gate_up(self):
        """Replace Gate and Up projections with one concatenated matmul."""
        if self.gate_up_proj is not None:
            return False
        self.gate_up_proj = FusedLinear.from_linears(
            (self.gate_proj, self.up_proj), backend=self.linear_backend
        )
        del self.gate_proj
        del self.up_proj
        return True

    def forward(self, x):
        if self.gate_up_proj is None:
            gate, up = self.gate_proj(x), self.up_proj(x)
        else:
            fused = self.gate_up_proj(x)
            # The two halves of a chunk are strided views. The custom CUDA
            # SiLU*Up kernel intentionally requires contiguous inputs.
            gate, up = (part.contiguous() for part in fused.chunk(2, dim=-1))
        return self.down_proj(silu_and_mul(gate, up, backend=self.elementwise_backend))


class DecoderLayer(nn.Module):
    def __init__(self, config, elementwise_backend="torch", data_backend="torch", linear_backend="torch"):
        super().__init__()
        self.elementwise_backend = elementwise_backend
        self.self_attn = Attention(config, elementwise_backend, data_backend, linear_backend)
        self.mlp = MLP(config, elementwise_backend, linear_backend)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, elementwise_backend)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps, elementwise_backend)

    def forward(self, x, *args, prefill_batch=None):
        attention = self.self_attn(self.input_layernorm(x), *args, prefill_batch=prefill_batch)
        normalized, summed = residual_rms_norm(
            attention, x, self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps, backend=self.elementwise_backend,
        )
        return residual_add(summed, self.mlp(normalized), backend=self.elementwise_backend)

    def forward_decode_batch(self, x, cos, sin, cache, block_tables, positions, layer, decode_backend):
        attention = self.self_attn.forward_decode_batch(
            self.input_layernorm(x), cos, sin, cache, block_tables, positions, layer, decode_backend
        )
        normalized, summed = residual_rms_norm(
            attention, x, self.post_attention_layernorm.weight,
            self.post_attention_layernorm.eps, backend=self.elementwise_backend,
        )
        return residual_add(summed, self.mlp(normalized), backend=self.elementwise_backend)


class Qwen3Backbone(nn.Module):
    def __init__(self, config, elementwise_backend="torch", data_backend="torch", linear_backend="torch"):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([DecoderLayer(config, elementwise_backend, data_backend, linear_backend) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps, elementwise_backend)


class NativeQwen3(nn.Module):
    def __init__(self, config: Qwen3Config, prefill_backend="auto", decode_backend="torch", elementwise_backend="torch", data_backend="torch", linear_backend="torch"):
        super().__init__()
        self.config = config
        if prefill_backend not in {"auto", "flash", "math", "cuda", "dandan"}:
            raise ValueError("unknown prefill backend")
        self.prefill_backend = prefill_backend
        if decode_backend not in {"torch", "cuda"}:
            raise ValueError("decode backend must be torch or cuda")
        self.decode_backend = decode_backend
        if elementwise_backend not in {"torch", "cuda"}:
            raise ValueError("elementwise backend must be torch or cuda")
        self.elementwise_backend = elementwise_backend
        if data_backend not in {"torch", "cuda"}:
            raise ValueError("data backend must be torch or cuda")
        self.data_backend = data_backend
        if linear_backend not in LINEAR_BACKENDS:
            raise ValueError("unknown linear backend")
        self.linear_backend = linear_backend
        self.quantization_report = None
        self.projection_fusion_enabled = False
        self.model = Qwen3Backbone(config, elementwise_backend, data_backend, linear_backend)
        self.lm_head = make_linear(config.hidden_size, config.vocab_size, bias=False, backend=linear_backend)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

    @classmethod
    def from_hf_model(cls, reference, prefill_backend="auto", decode_backend="torch", elementwise_backend="torch", data_backend="torch", linear_backend="torch"):
        config = Qwen3Config.from_hf(reference.config)
        # Reuse loaded storage instead of allocating a second 0.6B weight copy.
        with torch.device("meta"):
            model = cls(config, prefill_backend, decode_backend, elementwise_backend, data_backend, linear_backend)
        model.load_state_dict(reference.state_dict(), strict=True, assign=True)
        return model.eval()

    def quantize(self, config, *, backend="reference"):
        if backend == "dandan":
            from ..quantization.dandan import quantize_dandan_linears
            return quantize_dandan_linears(self, config)
        if backend != "reference":
            raise ValueError("quant backend must be reference or dandan")
        if self.projection_fusion_enabled:
            raise ValueError("quantization and projection fusion cannot be combined yet")
        from ..quantization.linear import quantize_model_linears
        return quantize_model_linears(self, config)

    def backend_report(self):
        report = {"linear_backend": self.linear_backend, "prefill_backend": self.prefill_backend,
                  "decode_backend": self.decode_backend, "projection_fusion": self.projection_fusion_enabled,
                  "quantization": self.quantization_report}
        if (self.linear_backend in DANDAN_LINEAR_BACKENDS or self.prefill_backend == "dandan"
                or (self.quantization_report or {}).get("accelerated_quant_gemm")):
            from ..kernels.dandan_backend import dependency_report
            report["dandan_ops"] = dependency_report()
        return report

    def enable_projection_fusion(self):
        """Fuse QKV and Gate/Up projections after HF weights are loaded.

        Quantized reference modules intentionally remain unfused for now so
        their per-format packing and state reports stay explicit.
        """
        if self.quantization_report is not None:
            raise ValueError("projection fusion is only available before quantization")
        changed = 0
        for layer in self.model.layers:
            changed += int(layer.self_attn.fuse_qkv())
            changed += int(layer.mlp.fuse_gate_up())
        self.projection_fusion_enabled = True
        return changed

    def create_cache(self, num_blocks, block_size):
        weight = self.model.embed_tokens.weight
        cfg = self.config
        return KVCachePool(cfg.num_hidden_layers, num_blocks, block_size, cfg.num_key_value_heads,
                           cfg.head_dim, dtype=weight.dtype, device=weight.device)

    @torch.inference_mode()
    def forward(self, input_ids, cache, block_ids=None, position=0, *, prefill_batch=None, logits_rows=None):
        """Unpadded prompt/suffix, or one paged decode token at any position.

        The caller must allocate/grow block_ids before calling. Position is the
        number of tokens already written, not the number of sampled outputs.
        """
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 1 or input_ids.numel() == 0 or input_ids.dtype != torch.long:
            raise ValueError("input_ids must be a nonempty 1D int64 tensor")
        if not isinstance(position, int) or isinstance(position, bool) or position < 0:
            raise ValueError("position must be a nonnegative integer")
        if prefill_batch is None and position + input_ids.numel() > self.config.max_position_embeddings:
            raise ValueError("request exceeds model context limit")
        cfg, shape = self.config, cache.shape
        if (shape.num_layers, shape.num_kv_heads, shape.head_dim) != (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim):
            raise ValueError("KV cache architecture mismatch")
        weight = self.model.embed_tokens.weight
        if cache.device != weight.device or cache.dtype != weight.dtype:
            raise ValueError("cache must match model device and dtype")
        if prefill_batch is not None or logits_rows is not None or position == 0 or input_ids.numel() > 1:
            from ..engine.batch import PrefillBatch

            if prefill_batch is None:
                if block_ids is None:
                    raise ValueError("prefill requires a block table")
                prefill_batch = PrefillBatch.from_sequences([input_ids], [block_ids], [position],
                    block_size=shape.block_size, num_blocks=shape.num_blocks, device=weight.device)
            elif not isinstance(prefill_batch, PrefillBatch) or input_ids is not prefill_batch.input_ids:
                raise ValueError("packed forward requires its PrefillBatch input_ids")
            return self._forward_prefill_batch(prefill_batch, cache, logits_rows=logits_rows)
        x = embedding_lookup(weight, input_ids.to(weight.device), backend=self.data_backend, validate=False)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=x.device).float() / cfg.head_dim))
        positions = torch.arange(position, position + x.shape[0], device=x.device).float()
        angles = positions[:, None] * inv_freq[None, :]
        cos, sin = angles.cos().to(x.dtype).unsqueeze(1), angles.sin().to(x.dtype).unsqueeze(1)
        for index, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, block_ids, position, index, self.prefill_backend, self.decode_backend)
        return self.lm_head(self.model.norm(x))

    def forward_prefill_batch(self, batch, cache, *, logits_rows=None):
        """Run packed query chunks while preserving Module forward hooks.

        CUDA prefill reads paged history directly. Dense flash/dandan and Torch
        paths retain per-request attention with shared packed projections/MLP.
        ``logits_rows`` selects packed query rows, preserving order/duplicates.
        """
        from ..engine.batch import PrefillBatch

        if not isinstance(batch, PrefillBatch):
            raise ValueError("batch must be a PrefillBatch")
        return self(batch.input_ids, cache, prefill_batch=batch, logits_rows=logits_rows)

    def _forward_prefill_batch(self, batch, cache, *, logits_rows):
        cfg, weight = self.config, self.model.embed_tokens.weight
        batch.validate(block_size=cache.shape.block_size, num_blocks=cache.shape.num_blocks,
            device=weight.device, max_position_embeddings=cfg.max_position_embeddings, vocab_size=cfg.vocab_size)
        rows = None
        if logits_rows is not None:
            from ..engine.batch import _integers

            row_values = _integers(logits_rows, "logits_rows")
            if any(row < 0 or row >= batch.num_tokens for row in row_values):
                raise ValueError("logits_rows index is outside the packed query range")
            rows = torch.tensor(row_values, device=weight.device, dtype=torch.int64)
        x = embedding_lookup(weight, batch.input_ids, backend=self.data_backend, validate=False)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=x.device).float() / cfg.head_dim))
        angles = batch.positions.to(dtype=torch.float32)[:, None] * inv_freq[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        for index, layer in enumerate(self.model.layers):
            x = layer(x, cos, sin, cache, None, 0, index, self.prefill_backend, self.decode_backend,
                      prefill_batch=batch)
        if rows is not None:
            if rows.numel() == 0:
                return weight.new_empty((0, cfg.vocab_size))
            x = x.index_select(0, rows)
        return self.lm_head(self.model.norm(x))

    @torch.inference_mode()
    def forward_decode_batch(self, input_ids, cache, block_tables, positions):
        """Decode one token per request using one batched model step."""
        if input_ids.ndim != 1 or input_ids.dtype != torch.long or input_ids.numel() == 0:
            raise ValueError("input_ids must be a nonempty 1D int64 batch")
        if positions.ndim != 1 or positions.shape != input_ids.shape or positions.dtype != torch.long:
            raise ValueError("positions must be int64 with one value per request")
        if block_tables.ndim != 2 or block_tables.shape[0] != input_ids.shape[0]:
            raise ValueError("block_tables must be [batch, max_blocks]")
        # GPU-to-CPU metadata checks are useful on the public eager path but
        # are illegal during CUDA Graph capture. The graph bucket owns fixed
        # trusted buffers, so defer this validation until replay is complete.
        # This check is intentionally kept in Python because it is a cheap
        # local status query in PyTorch's CUDA wrapper. Avoid any tensor
        # reduction or `.item()` while capture is active.
        capturing = bool(input_ids.is_cuda and torch.cuda.is_current_stream_capturing())
        if not capturing and (bool((positions < 0).any()) or bool((positions + 1 > self.config.max_position_embeddings).any())):
            raise ValueError("decode positions exceed model context")
        cfg, shape = self.config, cache.shape
        if (shape.num_layers, shape.num_kv_heads, shape.head_dim) != (cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim):
            raise ValueError("KV cache architecture mismatch")
        weight = self.model.embed_tokens.weight
        if cache.device != weight.device or cache.dtype != weight.dtype:
            raise ValueError("cache must match model device and dtype")
        x = embedding_lookup(weight, input_ids.to(weight.device), backend=self.data_backend, validate=False)
        inv_freq = 1.0 / (cfg.rope_theta ** (torch.arange(0, cfg.head_dim, 2, device=x.device).float() / cfg.head_dim))
        angles = positions.to(device=x.device, dtype=torch.float32)[:, None] * inv_freq[None, :]
        cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)
        for index, layer in enumerate(self.model.layers):
            x = layer.forward_decode_batch(x, cos, sin, cache, block_tables, positions, index, self.decode_backend)
        return self.lm_head(self.model.norm(x))
