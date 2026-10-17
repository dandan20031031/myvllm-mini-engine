from __future__ import annotations

from dataclasses import dataclass
import time

import torch

from ..sampling import sample_next_token


@dataclass
class IncrementalGenerationResult:
    text: str
    token_ids: list[int]
    finish_reason: str
    ttft_ms: float
    tpot_ms: float
    total_latency_ms: float


@dataclass
class Qwen3ModelRunner:
    """Optional Transformers-backed Qwen3 runner.

    Keeping this adapter small makes it possible to replace the model forward
    with a native implementation later while the engine and scheduler remain
    unchanged.
    """

    model_name_or_path: str
    device: str = "cuda"
    dtype: torch.dtype = torch.bfloat16
    enable_thinking: bool = False

    def __post_init__(self):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise ImportError("Qwen3 loading requires: pip install '.[model]'") from exc
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name_or_path,
            torch_dtype=self.dtype,
        )
        self.model.to(self.device)
        self.model.eval()

    def encode(self, prompt: str) -> list[int]:
        return self.tokenizer(prompt, add_special_tokens=True, return_tensors=None)["input_ids"]

    def _prompt_tensor(self, prompt: str) -> torch.Tensor:
        """Build the same generation prompt used by the chat API."""
        encoded = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=self.enable_thinking,
            return_tensors="pt",
        )
        return encoded.to(self.model.device)

    @torch.inference_mode()
    def logits(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.model(input_ids=input_ids, use_cache=True, **kwargs).logits[:, -1, :]

    @torch.inference_mode()
    def generate(self, prompt: str, max_new_tokens: int = 128, **kwargs) -> str:
        """Reference generation path used to validate engine outputs."""
        encoded = self._prompt_tensor(prompt)
        output = self.model.generate(encoded, max_new_tokens=max_new_tokens, **kwargs)
        generated = output[0, encoded.shape[-1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True)

    @torch.inference_mode()
    def generate_incremental(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        *,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
    ) -> IncrementalGenerationResult:
        """Reference prefill + one-token-at-a-time decode loop.

        This deliberately uses Transformers' ``past_key_values`` so it is a
        correctness baseline for the native paged-cache path.  The public
        result includes timing fields that the benchmark adapter can reuse.
        """
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        input_ids = self._prompt_tensor(prompt)
        attention_mask = torch.ones_like(input_ids)
        if input_ids.device.type == "cuda":
            torch.cuda.synchronize(input_ids.device)
        started = time.perf_counter()
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        past = outputs.past_key_values
        generated: list[int] = []
        eos = getattr(self.model.generation_config, "eos_token_id", None)
        if eos is None:
            eos = self.tokenizer.eos_token_id
        eos_ids = {int(eos)} if isinstance(eos, int) else {int(item) for item in (eos or [])}
        next_logits = outputs.logits[:, -1, :]
        first_token_time = None
        for index in range(max_new_tokens):
            token = sample_next_token(next_logits, temperature, top_k, top_p)
            token_id = int(token.item())
            generated.append(token_id)
            if first_token_time is None:
                first_token_time = time.perf_counter()
            if token_id in eos_ids or index + 1 == max_new_tokens:
                break
            input_ids = token.view(1, 1).to(self.model.device)
            attention_mask = torch.cat(
                (attention_mask, torch.ones((1, 1), dtype=attention_mask.dtype, device=attention_mask.device)),
                dim=1,
            )
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past,
                use_cache=True,
                return_dict=True,
            )
            past = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
        finished = time.perf_counter()
        ttft_ms = ((first_token_time or finished) - started) * 1000.0
        decode_count = len(generated) - 1
        tpot_ms = (finished - first_token_time) * 1000.0 / decode_count if decode_count else 0.0
        reason = "stop" if generated and generated[-1] in eos_ids else "length"
        return IncrementalGenerationResult(
            text=self.tokenizer.decode(generated, skip_special_tokens=True),
            token_ids=generated,
            finish_reason=reason,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            total_latency_ms=(finished - started) * 1000.0,
        )
