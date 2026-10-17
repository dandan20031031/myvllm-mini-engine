#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <vector>

at::Tensor paged_attention_cuda(const at::Tensor& query, const at::Tensor& key,
    const at::Tensor& value, const at::Tensor& tables, const at::Tensor& lengths,
    int64_t block_size);

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
    m.def("rms_norm", &rms_norm_checked, "FP16/BF16 RMSNorm");
    m.def("residual_rms_norm", &residual_rms_norm_checked, "FP16/BF16 residual RMSNorm");
    m.def("silu_and_mul", &silu_and_mul_checked, "FP16/BF16 fused SiLU multiply");
    m.def("residual_add", &residual_add_checked, "FP16/BF16 residual add");
}
