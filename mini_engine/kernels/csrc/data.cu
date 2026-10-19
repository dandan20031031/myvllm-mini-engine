// Windows CUDA data movement kernels for Qwen3.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {
constexpr int THREADS = 256;

template <typename scalar_t>
__global__ void embedding_kernel(const scalar_t* weight, const int64_t* ids, scalar_t* output,
                                 int64_t tokens, int64_t vocab, int64_t hidden) {
    const int64_t token = blockIdx.x;
    if (token >= tokens) return;
    const int64_t id = ids[token];
    if (id < 0 || id >= vocab) return;
    for (int64_t d = threadIdx.x; d < hidden; d += THREADS)
        output[token * hidden + d] = weight[id * hidden + d];
}

template <typename scalar_t>
__global__ void rope_kernel(const scalar_t* q, const scalar_t* k, const scalar_t* cos,
                            const scalar_t* sin, scalar_t* qo, scalar_t* ko,
                            int64_t tokens, int64_t q_heads, int64_t kv_heads, int64_t dim) {
    const int64_t token_head = blockIdx.x;
    const int64_t token = token_head / q_heads;
    const int64_t head = token_head % q_heads;
    const int64_t kv_head = head < kv_heads ? head : head % kv_heads;
    if (token >= tokens) return;
    const int64_t half = dim / 2;
    for (int64_t d = threadIdx.x; d < half; d += THREADS) {
        const float c = static_cast<float>(cos[token * half + d]);
        const float s = static_cast<float>(sin[token * half + d]);
        const int64_t qbase = (token * q_heads + head) * dim;
        const int64_t kbase = (token * kv_heads + kv_head) * dim;
        const float q1 = static_cast<float>(q[qbase + d]);
        const float q2 = static_cast<float>(q[qbase + half + d]);
        // Match eager FP16/BF16 semantics: each multiply is rounded before
        // the add/subtract, rather than rounding only the final expression.
        scalar_t qprod1 = static_cast<scalar_t>(q1 * c);
        scalar_t qprod2 = static_cast<scalar_t>(q2 * s);
        scalar_t qprod3 = static_cast<scalar_t>(q1 * s);
        scalar_t qprod4 = static_cast<scalar_t>(q2 * c);
        qo[qbase + d] = static_cast<scalar_t>(static_cast<float>(qprod1) - static_cast<float>(qprod2));
        qo[qbase + half + d] = static_cast<scalar_t>(static_cast<float>(qprod3) + static_cast<float>(qprod4));
        if (head < kv_heads) {
            const float k1 = static_cast<float>(k[kbase + d]);
            const float k2 = static_cast<float>(k[kbase + half + d]);
            scalar_t kprod1 = static_cast<scalar_t>(k1 * c);
            scalar_t kprod2 = static_cast<scalar_t>(k2 * s);
            scalar_t kprod3 = static_cast<scalar_t>(k1 * s);
            scalar_t kprod4 = static_cast<scalar_t>(k2 * c);
            ko[kbase + d] = static_cast<scalar_t>(static_cast<float>(kprod1) - static_cast<float>(kprod2));
            ko[kbase + half + d] = static_cast<scalar_t>(static_cast<float>(kprod3) + static_cast<float>(kprod4));
        }
    }
}

template <typename scalar_t>
__global__ void kv_write_kernel(scalar_t* key_cache, scalar_t* value_cache, const int64_t* ids,
                                const scalar_t* key, const scalar_t* value, int64_t tokens,
                                int64_t offset, int64_t block_size, int64_t blocks,
                                int64_t kv_heads, int64_t dim) {
    const int64_t token = blockIdx.x;
    if (token >= tokens) return;
    const int64_t logical = offset + token;
    const int64_t block_index = logical / block_size;
    const int64_t in_block = logical % block_size;
    if (block_index < 0 || block_index >= blocks) return;
    const int64_t physical = ids[block_index];
    if (physical < 0) return;
    const int64_t dst = ((physical * block_size + in_block) * kv_heads) * dim;
    const int64_t src = token * kv_heads * dim;
    for (int64_t d = threadIdx.x; d < kv_heads * dim; d += THREADS) {
        key_cache[dst + d] = key[src + d];
        value_cache[dst + d] = value[src + d];
    }
}
} // namespace

template <typename scalar_t>
void launch_embedding(const at::Tensor& weight, const at::Tensor& ids, at::Tensor& output, cudaStream_t stream) {
    embedding_kernel<scalar_t><<<static_cast<unsigned int>(ids.numel()), THREADS, 0, stream>>>(
        weight.data_ptr<scalar_t>(), ids.data_ptr<int64_t>(), output.data_ptr<scalar_t>(),
        ids.numel(), weight.size(0), weight.size(1));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_rope(const at::Tensor& q, const at::Tensor& k, const at::Tensor& cos, const at::Tensor& sin,
                 at::Tensor& qo, at::Tensor& ko, cudaStream_t stream) {
    rope_kernel<scalar_t><<<static_cast<unsigned int>(q.size(0) * q.size(1)), THREADS, 0, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), cos.data_ptr<scalar_t>(), sin.data_ptr<scalar_t>(),
        qo.data_ptr<scalar_t>(), ko.data_ptr<scalar_t>(), q.size(0), q.size(1), k.size(1), q.size(2));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void launch_write(const at::Tensor& kc, const at::Tensor& vc, const at::Tensor& ids,
                  int64_t offset, const at::Tensor& k, const at::Tensor& v, cudaStream_t stream) {
    kv_write_kernel<scalar_t><<<static_cast<unsigned int>(k.size(0)), THREADS, 0, stream>>>(
        kc.data_ptr<scalar_t>(), vc.data_ptr<scalar_t>(), ids.data_ptr<int64_t>(),
        k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), k.size(0), offset, kc.size(1), ids.numel(), kc.size(2), kc.size(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor embedding_lookup_cuda(const at::Tensor& weight, const at::Tensor& ids, bool validate) {
    const c10::cuda::CUDAGuard guard(weight.device());
    auto output = at::empty({ids.numel(), weight.size(1)}, weight.options());
    if (ids.numel() == 0) return output;
    if (validate && (!((ids >= 0) & (ids < weight.size(0))).all().item<bool>()))
        TORCH_CHECK(false, "embedding id out of range");
    auto stream = c10::cuda::getCurrentCUDAStream(weight.get_device()).stream();
    if (weight.scalar_type() == at::kHalf) launch_embedding<at::Half>(weight, ids, output, stream);
    else if (weight.scalar_type() == at::kBFloat16) launch_embedding<at::BFloat16>(weight, ids, output, stream);
    else launch_embedding<float>(weight, ids, output, stream);
    return output;
}

std::vector<at::Tensor> apply_rope_cuda(const at::Tensor& q, const at::Tensor& k,
                                        const at::Tensor& cos, const at::Tensor& sin, bool validate) {
    const c10::cuda::CUDAGuard guard(q.device());
    auto qo = at::empty_like(q), ko = at::empty_like(k);
    if (q.numel() == 0) return {qo, ko};
    if (validate) {
        TORCH_CHECK(q.size(2) % 2 == 0 && k.size(2) == q.size(2), "RoPE head dimension mismatch");
        TORCH_CHECK(cos.sizes() == at::IntArrayRef({q.size(0), q.size(2) / 2}), "RoPE cos shape mismatch");
        TORCH_CHECK(sin.sizes() == cos.sizes(), "RoPE sin shape mismatch");
    }
    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (q.scalar_type() == at::kHalf) launch_rope<at::Half>(q, k, cos, sin, qo, ko, stream);
    else launch_rope<at::BFloat16>(q, k, cos, sin, qo, ko, stream);
    return {qo, ko};
}

void write_kv_cache_cuda(const at::Tensor& kc, const at::Tensor& vc, const at::Tensor& ids,
                         int64_t offset, const at::Tensor& k, const at::Tensor& v,
                         int64_t block_size, bool validate) {
    const c10::cuda::CUDAGuard guard(kc.device());
    if (validate) TORCH_CHECK(!((ids < 0) | (ids >= kc.size(0))).any().item<bool>(), "KV physical block id out of range");
    auto stream = c10::cuda::getCurrentCUDAStream(kc.get_device()).stream();
    if (kc.scalar_type() == at::kHalf) launch_write<at::Half>(kc, vc, ids, offset, k, v, stream);
    else launch_write<at::BFloat16>(kc, vc, ids, offset, k, v, stream);
}
