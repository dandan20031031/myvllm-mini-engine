import torch


def sample_next_token(
    logits: torch.Tensor,
    temperature: float = 0.0,
    top_k: int = 0,
    top_p: float = 1.0,
    *, backend: str = "torch",
) -> torch.Tensor:
    """Reference sampler; temperature 0 selects greedy tokens."""
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, vocab]")
    if backend == "cuda":
        if temperature != 0:
            raise ValueError("CUDA sampling currently supports greedy temperature=0 only")
        from .kernels.cuda_extension import load_cuda_extension
        return load_cuda_extension().greedy_argmax(logits)
    if backend != "torch":
        raise ValueError("sampling backend must be torch or cuda")
    if temperature <= 0:
        return logits.argmax(dim=-1)
    scores = logits / temperature
    if top_k > 0:
        values, _ = torch.topk(scores, min(top_k, scores.shape[-1]), dim=-1)
        scores = scores.masked_fill(scores < values[:, -1:], float("-inf"))
    if top_p < 1.0:
        sorted_scores, sorted_indices = torch.sort(scores, descending=True, dim=-1)
        cumulative = torch.softmax(sorted_scores, dim=-1).cumsum(dim=-1)
        remove = cumulative - torch.softmax(sorted_scores, dim=-1) > top_p
        sorted_scores = sorted_scores.masked_fill(remove, float("-inf"))
        scores = torch.full_like(scores, float("-inf")).scatter(1, sorted_indices, sorted_scores)
    return torch.multinomial(torch.softmax(scores, dim=-1), num_samples=1).squeeze(-1)
