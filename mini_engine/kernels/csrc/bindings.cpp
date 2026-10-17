#include <torch/extension.h>

at::Tensor paged_attention_cuda(const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& tables, const at::Tensor& lengths,
    int64_t block_size);

at::Tensor paged_attention(const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& tables, const at::Tensor& lengths,
    int64_t block_size, bool validate) {
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
    return paged_attention_cuda(query, key, value, tables, lengths, block_size);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("paged_attention", &paged_attention, "FP16/BF16 split-KV paged decode attention");
}
