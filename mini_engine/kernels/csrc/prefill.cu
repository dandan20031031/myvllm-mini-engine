// Correctness-first causal prefill attention for Windows CUDA.
// One CTA computes one (batch, query_head, query_position) row. K/V are GQA.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>

namespace {
constexpr int THREADS = 128;

template <typename scalar_t>
__global__ void causal_prefill_kernel(
    const scalar_t* q, const scalar_t* k, const scalar_t* v, scalar_t* out,
    int64_t batch, int64_t q_heads, int64_t kv_heads, int64_t tokens, int64_t dim) {
    const int64_t row = blockIdx.x;
    const int64_t rows = batch * q_heads * tokens;
    if (row >= rows) return;
    const int64_t tq = row % tokens;
    const int64_t qh = (row / tokens) % q_heads;
    const int64_t b = row / (q_heads * tokens);
    const int64_t kvh = qh / (q_heads / kv_heads);
    extern __shared__ float scores[];
    float* meta = scores + tokens;
    const float scale = rsqrtf(static_cast<float>(dim));
    for (int64_t tk = threadIdx.x; tk <= tq; tk += THREADS) {
        float dot = 0.0f;
        const int64_t qbase = ((b * q_heads + qh) * tokens + tq) * dim;
        const int64_t kbase = ((b * kv_heads + kvh) * tokens + tk) * dim;
        for (int64_t d = 0; d < dim; ++d)
            dot += static_cast<float>(q[qbase + d]) * static_cast<float>(k[kbase + d]);
        scores[tk] = dot * scale;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = -INFINITY;
        for (int64_t tk = 0; tk <= tq; ++tk) maximum = fmaxf(maximum, scores[tk]);
        float denominator = 0.0f;
        for (int64_t tk = 0; tk <= tq; ++tk) {
            scores[tk] = expf(scores[tk] - maximum);
            denominator += scores[tk];
        }
        meta[0] = maximum;
        meta[1] = denominator;
    }
    __syncthreads();
    const float denominator = meta[1];
    const int64_t outbase = ((b * q_heads + qh) * tokens + tq) * dim;
    for (int64_t d = threadIdx.x; d < dim; d += THREADS) {
        float value = 0.0f;
        for (int64_t tk = 0; tk <= tq; ++tk) {
            const int64_t vbase = ((b * kv_heads + kvh) * tokens + tk) * dim;
            value += scores[tk] * static_cast<float>(v[vbase + d]);
        }
        out[outbase + d] = static_cast<scalar_t>(value / denominator);
    }
}
} // namespace

template <typename scalar_t>
void launch_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                    at::Tensor& output, cudaStream_t stream) {
    const int64_t rows = q.size(0) * q.size(1) * q.size(2);
    const size_t shared = static_cast<size_t>(q.size(2) + 2) * sizeof(float);
    causal_prefill_kernel<scalar_t><<<static_cast<unsigned int>(rows), THREADS, shared, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        q.size(0), q.size(1), k.size(1), q.size(2), q.size(3));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

at::Tensor causal_prefill_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    const c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "prefill tensors must be CUDA");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "prefill tensors must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes(), "invalid prefill tensor shapes");
    TORCH_CHECK(q.size(0) == k.size(0) && q.size(2) == k.size(2) && q.size(3) == k.size(3), "prefill shape mismatch");
    TORCH_CHECK(q.size(1) % k.size(1) == 0, "prefill GQA mismatch");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "prefill requires FP16/BF16");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(), "prefill dtype mismatch");
    TORCH_CHECK(q.size(2) > 0 && q.size(2) <= 8192, "CUDA prefill supports 1..8192 tokens; use SDPA for longer prompts");
    auto output = at::empty_like(q);
    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (q.scalar_type() == at::kHalf) launch_prefill<at::Half>(q, k, v, output, stream);
    else launch_prefill<at::BFloat16>(q, k, v, output, stream);
    return output;
}
