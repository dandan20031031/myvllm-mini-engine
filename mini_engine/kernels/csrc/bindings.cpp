#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <vector>

at::Tensor causal_prefill_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v);
at::Tensor causal_prefill_suffix_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int64_t query_start);

at::Tensor embedding_lookup_cuda(const at::Tensor& weight, const at::Tensor& ids, bool validate);
std::vector<at::Tensor> apply_rope_cuda(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& cos, const at::Tensor& sin, bool validate);
void write_kv_cache_cuda(const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& ids, int64_t offset, const at::Tensor& key, const at::Tensor& value,
    int64_t block_size, bool validate);
void write_kv_cache_batch_cuda(const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& block_tables, const at::Tensor& offsets, const at::Tensor& key, const at::Tensor& value,
    int64_t block_size, bool validate);

at::Tensor paged_attention_cuda(const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& tables, const at::Tensor& lengths,
    int64_t block_size, bool vectorized);

at::Tensor greedy_argmax_cuda(const at::Tensor& logits);

at::Tensor rms_norm_cuda(const at::Tensor& x, const at::Tensor& weight, double eps);
std::vector<at::Tensor> residual_rms_norm_cuda(const at::Tensor& x, const at::Tensor& residual,
    const at::Tensor& weight, double eps);
at::Tensor silu_and_mul_cuda(const at::Tensor& gate, const at::Tensor& up);

at::Tensor residual_add_cuda(const at::Tensor& x, const at::Tensor& residual);

void check_elementwise_tensor(const at::Tensor& x) {
    TORCH_CHECK(x.is_cuda(), "elementwise inputs must be CUDA");
    TORCH_CHECK(x.is_contiguous(), "elementwise inputs must be contiguous");
    TORCH_CHECK(x.scalar_type() == at::kHalf || x.scalar_type() == at::kBFloat16, "elementwise requires FP16/BF16");
    TORCH_CHECK(!at::GradMode::is_enabled() || !x.requires_grad(), "custom CUDA ops are inference-only; use inference_mode");
}

void check_pair(const at::Tensor& x, const at::Tensor& other) {
    check_elementwise_tensor(x);
    check_elementwise_tensor(other);
    TORCH_CHECK(x.device() == other.device(), "elementwise inputs must be on the same CUDA device");
    TORCH_CHECK(x.scalar_type() == other.scalar_type(), "elementwise dtype mismatch");
    TORCH_CHECK(x.sizes() == other.sizes(), "elementwise shape mismatch");
}

void check_norm(const at::Tensor& x, const at::Tensor& weight, double eps) {
    check_elementwise_tensor(x);
    check_elementwise_tensor(weight);
    TORCH_CHECK(x.dim() >= 1 && x.size(-1) > 0, "RMSNorm needs a nonempty last dimension");
    TORCH_CHECK(x.device() == weight.device(), "weight must be on the same CUDA device");
    TORCH_CHECK(weight.scalar_type() == x.scalar_type() && weight.dim() == 1 && weight.numel() == x.size(-1), "RMSNorm weight shape/dtype mismatch");
    TORCH_CHECK(std::isfinite(eps) && eps > 0 && eps <= std::numeric_limits<float>::max()
                && static_cast<float>(eps) > 0, "eps must be positive, finite and representable in FP32");
    TORCH_CHECK(x.numel() / x.size(-1) <= 2147483647, "too many RMSNorm rows");
}

at::Tensor rms_norm_checked(const at::Tensor& x, const at::Tensor& weight, double eps) {
    check_norm(x, weight, eps);
    return rms_norm_cuda(x, weight, eps);
}

std::vector<at::Tensor> residual_rms_norm_checked(const at::Tensor& x, const at::Tensor& residual,
    const at::Tensor& weight, double eps) {
    check_norm(x, weight, eps);
    check_pair(x, residual);
    return residual_rms_norm_cuda(x, residual, weight, eps);
}

at::Tensor residual_add_checked(const at::Tensor& x, const at::Tensor& residual) {
    check_pair(x, residual);
    return residual_add_cuda(x, residual);
}

at::Tensor silu_and_mul_checked(const at::Tensor& gate, const at::Tensor& up) {
    check_pair(gate, up);
    return silu_and_mul_cuda(gate, up);
}

at::Tensor embedding_lookup_checked(const at::Tensor& weight, const at::Tensor& ids, bool validate) {
    TORCH_CHECK(weight.is_cuda() && ids.is_cuda(), "embedding inputs must be CUDA");
    TORCH_CHECK(weight.is_contiguous() && ids.is_contiguous(), "embedding inputs must be contiguous");
    TORCH_CHECK(weight.dim() == 2 && ids.dim() == 1, "embedding expects weight [vocab, hidden] and ids [tokens]");
    TORCH_CHECK(ids.scalar_type() == at::kLong, "embedding ids must be int64");
    TORCH_CHECK(weight.scalar_type() == at::kHalf || weight.scalar_type() == at::kBFloat16 || weight.scalar_type() == at::kFloat,
                "embedding supports FP32/FP16/BF16");
    TORCH_CHECK(weight.size(0) > 0 && weight.size(1) > 0, "embedding weight must be nonempty");
    return embedding_lookup_cuda(weight, ids, validate);
}

std::vector<at::Tensor> apply_rope_checked(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& cos, const at::Tensor& sin, bool validate) {
    for (const auto& t : {q, k, cos, sin}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "RoPE inputs must be contiguous CUDA tensors");
        TORCH_CHECK(t.device() == q.device(), "RoPE inputs must share a device");
    }
    TORCH_CHECK(q.dim() == 3 && k.dim() == 3 && q.size(0) == k.size(0) && q.size(2) == k.size(2), "RoPE Q/K shape mismatch");
    TORCH_CHECK(cos.dim() == 2 && sin.sizes() == cos.sizes() && cos.size(0) == q.size(0) && cos.size(1) * 2 == q.size(2), "RoPE cos/sin shape mismatch");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "RoPE requires FP16/BF16 Q/K");
    TORCH_CHECK(cos.scalar_type() == q.scalar_type() && sin.scalar_type() == q.scalar_type(), "RoPE dtype mismatch");
    TORCH_CHECK(q.size(1) >= k.size(1) && q.size(1) % k.size(1) == 0, "RoPE GQA head mismatch");
    return apply_rope_cuda(q, k, cos, sin, validate);
}

void write_kv_cache_checked(const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& ids, int64_t offset, const at::Tensor& key, const at::Tensor& value,
    int64_t block_size, bool validate) {
    for (const auto& t : {key_cache, value_cache, ids, key, value}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "KV write inputs must be contiguous CUDA tensors");
        TORCH_CHECK(t.device() == key_cache.device(), "KV write inputs must share a device");
    }
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(), "cache shape mismatch");
    TORCH_CHECK(key.dim() == 3 && value.sizes() == key.sizes(), "K/V shape mismatch");
    TORCH_CHECK(ids.dim() == 1 && ids.scalar_type() == at::kLong, "block ids must be int64 [blocks]");
    TORCH_CHECK(block_size > 0 && key_cache.size(1) == block_size, "block size mismatch");
    TORCH_CHECK(key.size(1) == key_cache.size(2) && key.size(2) == key_cache.size(3), "K/V head shape mismatch");
    TORCH_CHECK(key_cache.scalar_type() == at::kHalf || key_cache.scalar_type() == at::kBFloat16, "KV cache requires FP16/BF16");
    TORCH_CHECK(key.scalar_type() == key_cache.scalar_type() && value.scalar_type() == key_cache.scalar_type(), "KV dtype mismatch");
    TORCH_CHECK(offset >= 0 && offset + key.size(0) <= ids.size(0) * block_size, "KV write exceeds block table");
    write_kv_cache_cuda(key_cache, value_cache, ids, offset, key, value, block_size, validate);
}

void write_kv_cache_batch_checked(const at::Tensor& key_cache, const at::Tensor& value_cache,
    const at::Tensor& tables, const at::Tensor& offsets, const at::Tensor& key, const at::Tensor& value,
    int64_t block_size, bool validate) {
    for (const auto& t : {key_cache, value_cache, tables, offsets, key, value}) {
        TORCH_CHECK(t.is_cuda() && t.is_contiguous(), "batch KV write inputs must be contiguous CUDA tensors");
        TORCH_CHECK(t.device() == key_cache.device(), "batch KV write inputs must share a device");
    }
    TORCH_CHECK(key_cache.dim() == 4 && value_cache.sizes() == key_cache.sizes(), "batch cache shape mismatch");
    TORCH_CHECK(key.dim() == 3 && value.sizes() == key.sizes(), "batch K/V shape mismatch");
    TORCH_CHECK(tables.dim() == 2 && offsets.dim() == 1 && tables.size(0) == key.size(0) && offsets.size(0) == key.size(0), "batch table shape mismatch");
    TORCH_CHECK(tables.scalar_type() == at::kLong && offsets.scalar_type() == at::kLong, "batch metadata must be int64");
    TORCH_CHECK(block_size > 0 && key_cache.size(1) == block_size, "batch block size mismatch");
    TORCH_CHECK(key.size(1) == key_cache.size(2) && key.size(2) == key_cache.size(3), "batch K/V shape mismatch");
    TORCH_CHECK(key_cache.scalar_type() == at::kHalf || key_cache.scalar_type() == at::kBFloat16, "batch KV cache requires FP16/BF16");
    TORCH_CHECK(key.scalar_type() == key_cache.scalar_type() && value.scalar_type() == key_cache.scalar_type(), "batch KV dtype mismatch");
    if (validate) {
        // This reduction synchronizes the host and is not legal while a CUDA
        // Graph is being captured. Trusted native graph inputs are already
        // bounded by the scheduler and kernel guards.
        TORCH_CHECK(!((offsets < 0) | (offsets >= tables.size(1) * block_size)).any().item<bool>(), "batch KV offset out of range");
    }
    write_kv_cache_batch_cuda(key_cache, value_cache, tables, offsets, key, value, block_size, validate);
}

at::Tensor paged_attention(const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& tables, const at::Tensor& lengths,
    int64_t block_size, bool validate, bool vectorized) {
    TORCH_CHECK(query.is_cuda(), "query must be CUDA");
    for (const auto& t : {key, value, tables, lengths}) {
        TORCH_CHECK(t.device() == query.device(), "all inputs must be on the same CUDA device");
        TORCH_CHECK(t.is_contiguous(), "inputs must be contiguous");
    }
    TORCH_CHECK(query.is_contiguous(), "query must be contiguous");
    TORCH_CHECK(query.dim() == 3 && key.dim() == 4 && value.sizes() == key.sizes(), "invalid query/cache shapes");
    TORCH_CHECK(query.scalar_type() == at::kHalf || query.scalar_type() == at::kBFloat16, "FP16/BF16 required");
    TORCH_CHECK(key.scalar_type() == query.scalar_type() && value.scalar_type() == query.scalar_type(), "cache dtype mismatch");
    TORCH_CHECK(tables.scalar_type() == at::kLong && lengths.scalar_type() == at::kLong, "metadata must be int64");
    TORCH_CHECK(query.size(0) > 0 && query.size(1) > 0 && key.size(0) > 0 && key.size(2) > 0, "empty batch/heads/cache");
    TORCH_CHECK(block_size > 0 && key.size(1) == block_size, "block size mismatch");
    TORCH_CHECK(query.size(2) == key.size(3) && query.size(1) % key.size(2) == 0, "head dimension or GQA mismatch");
    const auto dim = query.size(2);
    TORCH_CHECK(dim == 16 || dim == 32 || dim == 64 || dim == 80 || dim == 96 || dim == 128 || dim == 192 || dim == 256,
                "supported head_dim: 16,32,64,80,96,128,192,256");
    TORCH_CHECK(tables.dim() == 2 && tables.size(0) == query.size(0) && tables.size(1) > 0, "invalid block table");
    TORCH_CHECK(lengths.dim() == 1 && lengths.size(0) == query.size(0), "invalid context lengths");
    TORCH_CHECK(tables.size(1) * block_size <= 1048576, "maximum table capacity is 1M tokens");
    TORCH_CHECK(query.size(0) * query.size(1) <= 2147483647, "batch/head product too large");
    if (validate) {
        // Public calls validate GPU metadata. The trusted model path can skip
        // these synchronizing checks; kernels still guard physical indices.
        TORCH_CHECK(!((lengths <= 0) | (lengths > tables.size(1) * block_size)).any().item<bool>(), "invalid context length");
        auto positions = at::arange(tables.size(1), lengths.options()) * block_size;
        auto active = positions.unsqueeze(0) < lengths.unsqueeze(1);
        TORCH_CHECK(!(((tables < 0) | (tables >= key.size(0))) & active).any().item<bool>(), "physical block id out of range");
    }
    return paged_attention_cuda(query, key, value, tables, lengths, block_size, vectorized);
}

at::Tensor greedy_argmax_checked(const at::Tensor& logits) {
    TORCH_CHECK(logits.is_cuda() && logits.is_contiguous(), "greedy logits must be contiguous CUDA tensors");
    TORCH_CHECK(logits.dim()==2 && logits.size(1)>0, "greedy logits must be [batch, nonempty vocab]");
    TORCH_CHECK(logits.scalar_type()==at::kFloat || logits.scalar_type()==at::kHalf || logits.scalar_type()==at::kBFloat16,
                "greedy supports FP32/FP16/BF16");
    TORCH_CHECK(logits.size(0)<=65535 && logits.size(1)<=2147483647, "greedy logits shape too large");
    return greedy_argmax_cuda(logits);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention", [](const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
        const at::Tensor& t,const at::Tensor& l,int64_t b,bool validate) {
        return paged_attention(q,k,v,t,l,b,validate,true);
    }, "FP16/BF16 split-KV paged decode attention");
    m.def("paged_attention_baseline", [](const at::Tensor& q,const at::Tensor& k,const at::Tensor& v,
        const at::Tensor& t,const at::Tensor& l,int64_t b,bool validate) {
        return paged_attention(q,k,v,t,l,b,validate,false);
    }, "Scalar baseline for performance ablation");
    m.def("greedy_argmax", &greedy_argmax_checked, "Stable two-pass CUDA greedy sampling");
    m.def("rms_norm", &rms_norm_checked, "FP16/BF16 RMSNorm");
    m.def("residual_rms_norm", &residual_rms_norm_checked, "FP16/BF16 residual RMSNorm");
    m.def("silu_and_mul", &silu_and_mul_checked, "FP16/BF16 fused SiLU multiply");
    m.def("residual_add", &residual_add_checked, "FP16/BF16 residual add");
    m.def("embedding_lookup", &embedding_lookup_checked, "FP32/FP16/BF16 embedding gather");
    m.def("apply_rope", &apply_rope_checked, "FP16/BF16 rotary embedding");
    m.def("write_kv_cache", &write_kv_cache_checked, "FP16/BF16 paged KV scatter");
    m.def("write_kv_cache_batch", &write_kv_cache_batch_checked, "FP16/BF16 batched paged KV scatter");
    m.def("causal_prefill", &causal_prefill_cuda, "FP16/BF16 causal prefill attention");
    m.def("causal_prefill_suffix", &causal_prefill_suffix_cuda, "FP16/BF16 offset causal suffix attention");
}
