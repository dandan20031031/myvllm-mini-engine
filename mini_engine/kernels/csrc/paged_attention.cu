// Project-owned decode attention: one CTA per (request, query head, partition).
// Logical tokens address physical KV pages directly; no gather/repeat tensor.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <cmath>
#include <limits>

namespace {
constexpr int THREADS = 128;
constexpr int WARPS = THREADS / 32;
constexpr int PARTITION = 256;

inline void check_launch_capture_safe(cudaStream_t) {}

template <bool MAXIMUM>
__device__ float warp_reduce(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        float other = __shfl_down_sync(0xffffffff, v, offset);
        v = MAXIMUM ? fmaxf(v, other) : v + other;
    }
    return v;
}

template <bool MAXIMUM>
__device__ float block_reduce(float value, float* scratch) {
    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    value = warp_reduce<MAXIMUM>(value);
    if (lane == 0) scratch[warp] = value;
    __syncthreads();
    value = threadIdx.x < WARPS ? scratch[lane] : (MAXIMUM ? -INFINITY : 0.f);
    if (warp == 0) {
        value = warp_reduce<MAXIMUM>(value);
        if (lane == 0) scratch[0] = value;
    }
    __syncthreads();
    float result = scratch[0];
    __syncthreads(); // protect scratch readers from the next reduction
    return result;
}

template <typename scalar_t, int DIM>
__global__ void attention_partition(const scalar_t* query, const scalar_t* key,
    const scalar_t* value, const int64_t* tables, const int64_t* lengths,
    scalar_t* output, float* partial, float* lse, int heads, int kv_heads,
    int num_blocks, int block_size, int table_width, int partitions) {
    const int request = blockIdx.x / heads;
    const int head = blockIdx.x % heads;
    const int kv_head = head / (heads / kv_heads);
    const int part = blockIdx.y;
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    const int64_t row = static_cast<int64_t>(blockIdx.x) * partitions + part;
    const int length = static_cast<int>(lengths[request]);
    const int start = part * PARTITION;
    const int count = length > start && length <= table_width * block_size
                      ? min(PARTITION, length - start) : 0;
    __shared__ float scores[PARTITION];
    __shared__ float scratch[WARPS];

    float q[(DIM + 31) / 32];
    #pragma unroll
    for (int i = 0; i < (DIM + 31) / 32; ++i) {
        int d = lane + i * 32;
        q[i] = d < DIM ? static_cast<float>(query[static_cast<int64_t>(blockIdx.x) * DIM + d]) : 0.f;
    }
    for (int t = warp; t < count; t += WARPS) {
        const int token = start + t;
        const int64_t page = tables[static_cast<int64_t>(request) * table_width + token / block_size];
        const bool valid = page >= 0 && page < num_blocks;
        const int64_t base = ((page * block_size + token % block_size) * kv_heads + kv_head) * DIM;
        float score = 0.f;
        if (valid) {
            #pragma unroll
            for (int i = 0; i < (DIM + 31) / 32; ++i) {
                int d = lane + i * 32;
                if (d < DIM) score += q[i] * static_cast<float>(key[base + d]);
            }
        }
        score = warp_reduce<false>(score);
        if (lane == 0) scores[t] = valid ? score * rsqrtf(static_cast<float>(DIM)) : -INFINITY;
    }
    __syncthreads();
    float local_max = -INFINITY;
    for (int t = tid; t < count; t += THREADS) local_max = fmaxf(local_max, scores[t]);
    const float maximum = block_reduce<true>(local_max, scratch);
    float local_sum = 0.f;
    for (int t = tid; t < count; t += THREADS) {
        float probability = isfinite(maximum) ? expf(scores[t] - maximum) : 0.f;
        scores[t] = probability;
        local_sum += probability;
    }
    const float sum = block_reduce<false>(local_sum, scratch);
    if (tid == 0 && partitions > 1) lse[row] = sum > 0.f ? maximum + logf(sum) : -INFINITY;
    for (int d = tid; d < DIM; d += THREADS) {
        float acc = 0.f;
        for (int t = 0; t < count; ++t) {
            const int token = start + t;
            const int64_t page = tables[static_cast<int64_t>(request) * table_width + token / block_size];
            if (page >= 0 && page < num_blocks) {
                const int64_t base = ((page * block_size + token % block_size) * kv_heads + kv_head) * DIM;
                acc += scores[t] * static_cast<float>(value[base + d]);
            }
        }
        float result = sum > 0.f ? acc / sum : 0.f;
        if (partitions == 1) output[static_cast<int64_t>(blockIdx.x) * DIM + d] = static_cast<scalar_t>(result);
        else partial[row * DIM + d] = result;
    }
}

template <typename scalar_t, int DIM>
__global__ void merge_partitions(const float* partial, const float* lse,
                                 scalar_t* output, int partitions) {
    __shared__ float scratch[WARPS];
    const int tid = threadIdx.x;
    const int64_t base = static_cast<int64_t>(blockIdx.x) * partitions;
    float m = -INFINITY;
    for (int p = tid; p < partitions; p += THREADS) m = fmaxf(m, lse[base + p]);
    m = block_reduce<true>(m, scratch);
    float sum = 0.f;
    for (int p = tid; p < partitions; p += THREADS) sum += isfinite(m) ? expf(lse[base + p] - m) : 0.f;
    sum = block_reduce<false>(sum, scratch);
    for (int d = tid; d < DIM; d += THREADS) {
        float result = 0.f;
        if (sum > 0.f) {
            for (int p = 0; p < partitions; ++p) result += expf(lse[base + p] - m) * partial[(base + p) * DIM + d];
            result /= sum;
        }
        output[static_cast<int64_t>(blockIdx.x) * DIM + d] = static_cast<scalar_t>(result);
    }
}

template <typename scalar_t, int DIM>
void launch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
            const at::Tensor& tables, const at::Tensor& lengths, at::Tensor& out,
            at::Tensor& partial, at::Tensor& lse, int parts, cudaStream_t stream) {
    int rows = static_cast<int>(q.size(0) * q.size(1));
    attention_partition<scalar_t, DIM><<<dim3(rows, parts), THREADS, 0, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(),
        tables.data_ptr<int64_t>(), lengths.data_ptr<int64_t>(), out.data_ptr<scalar_t>(),
        parts > 1 ? partial.data_ptr<float>() : nullptr, parts > 1 ? lse.data_ptr<float>() : nullptr,
        q.size(1), k.size(2), k.size(0), k.size(1), tables.size(1), parts);
    check_launch_capture_safe(stream);
    if (parts > 1) {
        merge_partitions<scalar_t, DIM><<<rows, THREADS, 0, stream>>>(
            partial.data_ptr<float>(), lse.data_ptr<float>(), out.data_ptr<scalar_t>(), parts);
        check_launch_capture_safe(stream);
    }
}

template <typename scalar_t>
void dispatch(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
              const at::Tensor& tables, const at::Tensor& lengths, at::Tensor& out,
              at::Tensor& partial, at::Tensor& lse, int parts, cudaStream_t stream) {
    #define HEAD_CASE(D) case D: launch<scalar_t, D>(q,k,v,tables,lengths,out,partial,lse,parts,stream); break
    switch (q.size(2)) {
        HEAD_CASE(16); HEAD_CASE(32); HEAD_CASE(64); HEAD_CASE(80);
        HEAD_CASE(96); HEAD_CASE(128); HEAD_CASE(192); HEAD_CASE(256);
    }
    #undef HEAD_CASE
}
} // namespace

at::Tensor paged_attention_cuda(const at::Tensor& q, const at::Tensor& k,
    const at::Tensor& v, const at::Tensor& tables, const at::Tensor& lengths, int64_t block_size) {
    const c10::cuda::CUDAGuard guard(q.device());
    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    auto output = at::empty_like(q);
    const int parts = static_cast<int>((tables.size(1) * block_size + PARTITION - 1) / PARTITION);
    at::Tensor partial, lse;
    if (parts > 1) {
        partial = at::empty({q.size(0), q.size(1), parts, q.size(2)}, q.options().dtype(at::kFloat));
        lse = at::empty({q.size(0), q.size(1), parts}, q.options().dtype(at::kFloat));
    }
    if (q.scalar_type() == at::kHalf) dispatch<at::Half>(q,k,v,tables,lengths,output,partial,lse,parts,stream);
    else dispatch<at::BFloat16>(q,k,v,tables,lengths,output,partial,lse,parts,stream);
    return output;
}
