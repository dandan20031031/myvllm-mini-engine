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

inline void check_launch_capture_safe(cudaStream_t) { C10_CUDA_KERNEL_LAUNCH_CHECK(); }

template <typename scalar_t>
__global__ void causal_prefill_kernel(
    const scalar_t* q, const scalar_t* k, const scalar_t* v, scalar_t* out,
    int64_t batch, int64_t q_heads, int64_t kv_heads, int64_t tokens, int64_t dim,
    int64_t kv_tokens, int64_t query_start) {
    const int64_t row = blockIdx.x;
    const int64_t rows = batch * q_heads * tokens;
    if (row >= rows) return;
    const int64_t tq = row % tokens;
    const int64_t last_key = query_start + tq;
    const int64_t qh = (row / tokens) % q_heads;
    const int64_t b = row / (q_heads * tokens);
    const int64_t kvh = qh / (q_heads / kv_heads);
    extern __shared__ float scores[];
    float* meta = scores + kv_tokens;
    const float scale = rsqrtf(static_cast<float>(dim));
    for (int64_t tk = threadIdx.x; tk <= last_key; tk += THREADS) {
        float dot = 0.0f;
        const int64_t qbase = ((b * q_heads + qh) * tokens + tq) * dim;
        const int64_t kbase = ((b * kv_heads + kvh) * kv_tokens + tk) * dim;
        for (int64_t d = 0; d < dim; ++d)
            dot += static_cast<float>(q[qbase + d]) * static_cast<float>(k[kbase + d]);
        scores[tk] = dot * scale;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float maximum = -INFINITY;
        for (int64_t tk = 0; tk <= last_key; ++tk) maximum = fmaxf(maximum, scores[tk]);
        float denominator = 0.0f;
        for (int64_t tk = 0; tk <= last_key; ++tk) {
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
        for (int64_t tk = 0; tk <= last_key; ++tk) {
            const int64_t vbase = ((b * kv_heads + kvh) * kv_tokens + tk) * dim;
            value += scores[tk] * static_cast<float>(v[vbase + d]);
        }
        out[outbase + d] = static_cast<scalar_t>(value / denominator);
    }
}
} // namespace

template <typename scalar_t>
void launch_prefill(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
                    at::Tensor& output, cudaStream_t stream, int64_t query_start) {
    const int64_t rows = q.size(0) * q.size(1) * q.size(2);
    const size_t shared = static_cast<size_t>(k.size(2) + 2) * sizeof(float);
    causal_prefill_kernel<scalar_t><<<static_cast<unsigned int>(rows), THREADS, shared, stream>>>(
        q.data_ptr<scalar_t>(), k.data_ptr<scalar_t>(), v.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
        q.size(0), q.size(1), k.size(1), q.size(2), q.size(3), k.size(2), query_start);
    check_launch_capture_safe(stream);
}

at::Tensor causal_prefill_suffix_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v, int64_t query_start) {
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(), "prefill tensors must be CUDA");
    const c10::cuda::CUDAGuard guard(q.device());
    TORCH_CHECK(q.device()==k.device() && q.device()==v.device(), "prefill tensors must share device");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(), "prefill tensors must be contiguous");
    TORCH_CHECK(q.dim() == 4 && k.dim() == 4 && v.sizes() == k.sizes(), "invalid prefill tensor shapes");
    TORCH_CHECK(q.size(0)>0 && q.size(0) == k.size(0) && q.size(3)>0 && q.size(3) == k.size(3), "prefill shape mismatch");
    TORCH_CHECK(k.size(1)>0 && q.size(1)>0 && q.size(1) % k.size(1) == 0, "prefill GQA mismatch");
    TORCH_CHECK(query_start>=0 && q.size(2)>0 && query_start+q.size(2)<=k.size(2), "prefill query range exceeds KV sequence");
    TORCH_CHECK(q.scalar_type() == at::kHalf || q.scalar_type() == at::kBFloat16, "prefill requires FP16/BF16");
    TORCH_CHECK(k.scalar_type() == q.scalar_type() && v.scalar_type() == q.scalar_type(), "prefill dtype mismatch");
    TORCH_CHECK(k.size(2) > 0 && k.size(2) <= 8192, "CUDA prefill supports 1..8192 tokens; use SDPA for longer prompts");
    auto output = at::empty_like(q);
    auto stream = c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (q.scalar_type() == at::kHalf) launch_prefill<at::Half>(q, k, v, output, stream, query_start);
    else launch_prefill<at::BFloat16>(q, k, v, output, stream, query_start);
    return output;
}

at::Tensor causal_prefill_cuda(const at::Tensor& q, const at::Tensor& k, const at::Tensor& v) {
    TORCH_CHECK(q.dim()==4 && k.dim()==4 && q.size(2)==k.size(2), "square prefill requires equal Q/K lengths");
    return causal_prefill_suffix_cuda(q,k,v,0);
}

namespace {
constexpr int PAGED_CHUNK = 64;
constexpr int PAGED_DIM_LIMIT = 1024;

// One CTA per packed query/head. Online softmax retains only a 64-key tile
// and per-feature accumulators: shared memory does not grow with context.
template <typename scalar_t>
__global__ void paged_prefill_kernel(const scalar_t* q, const scalar_t* kc, const scalar_t* vc,
    const int64_t* offsets, const int64_t* starts, const int64_t* tables,
    const int64_t* lengths, scalar_t* output, int64_t tokens, int64_t q_heads,
    int64_t kv_heads, int64_t dim, int64_t requests, int64_t table_width,
    int64_t block_size, int64_t pages) {
    const int64_t row = blockIdx.x;
    const int64_t token = row / q_heads, qh = row % q_heads;
    if (token >= tokens) return;
    int64_t low=0, high=requests;
    while (low<high) {
        const int64_t middle=(low+high)/2;
        if (offsets[middle+1]<=token) low=middle+1;
        else high=middle;
    }
    if (low>=requests || token<offsets[low]) return;
    const int64_t last=starts[low]+token-offsets[low];
    if (last<0 || last>=lengths[low] || last/block_size>=table_width) return;
    const int64_t kvh=qh/(q_heads/kv_heads), qbase=row*dim;
    const float scale=rsqrtf(static_cast<float>(dim));
    __shared__ float weights[PAGED_CHUNK];
    __shared__ float state[3]; // running maximum, denominator, previous scale
    float accumulator[PAGED_DIM_LIMIT/THREADS];
    #pragma unroll
    for (int i=0; i<PAGED_DIM_LIMIT/THREADS; ++i) accumulator[i]=0.f;
    if (threadIdx.x==0) {state[0]=-INFINITY; state[1]=0.f;}
    __syncthreads();
    for (int64_t base=0; base<=last; base+=PAGED_CHUNK) {
        if (threadIdx.x<PAGED_CHUNK) {
            const int64_t key=base+threadIdx.x;
            float score=-INFINITY;
            if (key<=last) {
                const int64_t page=tables[low*table_width+key/block_size];
                if (page>=0 && page<pages) {
                    const int64_t kbase=((page*block_size+key%block_size)*kv_heads+kvh)*dim;
                    score=0.f;
                    for (int64_t d=0; d<dim; ++d)
                        score+=static_cast<float>(q[qbase+d])*static_cast<float>(kc[kbase+d]);
                    score*=scale;
                }
            }
            weights[threadIdx.x]=score;
        }
        __syncthreads();
        if (threadIdx.x==0) {
            float maximum=state[0];
            for (int i=0; i<PAGED_CHUNK && base+i<=last; ++i) maximum=fmaxf(maximum, weights[i]);
            const float alpha=expf(state[0]-maximum);
            float denominator=state[1]*alpha;
            for (int i=0; i<PAGED_CHUNK; ++i) {
                weights[i]=base+i<=last ? expf(weights[i]-maximum) : 0.f;
                denominator+=weights[i];
            }
            state[0]=maximum; state[1]=denominator; state[2]=alpha;
        }
        __syncthreads();
        #pragma unroll
        for (int i=0; i<PAGED_DIM_LIMIT/THREADS; ++i) {
            const int64_t d=threadIdx.x+i*THREADS;
            if (d<dim) {
                float weighted=0.f;
                for (int j=0; j<PAGED_CHUNK && base+j<=last; ++j) {
                    const int64_t key=base+j;
                    const int64_t page=tables[low*table_width+key/block_size];
                    if (page>=0 && page<pages) {
                        const int64_t vbase=((page*block_size+key%block_size)*kv_heads+kvh)*dim;
                        weighted+=weights[j]*static_cast<float>(vc[vbase+d]);
                    }
                }
                accumulator[i]=accumulator[i]*state[2]+weighted;
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i=0; i<PAGED_DIM_LIMIT/THREADS; ++i) {
        const int64_t d=threadIdx.x+i*THREADS;
        if (d<dim) output[qbase+d]=static_cast<scalar_t>(accumulator[i]/state[1]);
    }
}

template <typename scalar_t>
void launch_paged_prefill(const at::Tensor& q, const at::Tensor& kc, const at::Tensor& vc,
    const at::Tensor& offsets, const at::Tensor& starts, const at::Tensor& tables,
    const at::Tensor& lengths, at::Tensor& output, int64_t block_size, cudaStream_t stream) {
    paged_prefill_kernel<scalar_t><<<static_cast<unsigned int>(q.size(0)*q.size(1)), THREADS, 0, stream>>>(
        q.data_ptr<scalar_t>(), kc.data_ptr<scalar_t>(), vc.data_ptr<scalar_t>(),
        offsets.data_ptr<int64_t>(), starts.data_ptr<int64_t>(), tables.data_ptr<int64_t>(),
        lengths.data_ptr<int64_t>(), output.data_ptr<scalar_t>(), q.size(0), q.size(1),
        kc.size(2), q.size(2), starts.numel(), tables.size(1), block_size, kc.size(0));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}

at::Tensor paged_prefill_attention_cuda(const at::Tensor& q, const at::Tensor& kc,
    const at::Tensor& vc, const at::Tensor& offsets, const at::Tensor& starts,
    const at::Tensor& tables, const at::Tensor& lengths, int64_t block_size, bool validate) {
    TORCH_CHECK(q.is_cuda(), "paged prefill Q must be CUDA");
    const c10::cuda::CUDAGuard guard(q.device());
    for (const auto& t : {q, kc, vc, offsets, starts, tables, lengths})
        TORCH_CHECK(t.device()==q.device() && t.is_contiguous(), "paged prefill inputs must be contiguous on one CUDA device");
    TORCH_CHECK(q.dim()==3 && kc.dim()==4 && vc.sizes()==kc.sizes(), "paged prefill Q/cache shape mismatch");
    TORCH_CHECK(block_size>0 && kc.size(0)>0 && kc.size(1)==block_size && kc.size(2)>0 && kc.size(3)>0,
                "paged prefill cache/block size mismatch");
    TORCH_CHECK(q.size(1)>0 && q.size(1)%kc.size(2)==0 && q.size(2)==kc.size(3) && q.size(2)<=PAGED_DIM_LIMIT,
                "paged prefill GQA/head_dim mismatch (maximum head_dim 1024)");
    TORCH_CHECK(q.scalar_type()==at::kHalf || q.scalar_type()==at::kBFloat16, "paged prefill requires FP16/BF16");
    TORCH_CHECK(kc.scalar_type()==q.scalar_type() && vc.scalar_type()==q.scalar_type(), "paged prefill dtype mismatch");
    TORCH_CHECK(offsets.scalar_type()==at::kLong && starts.scalar_type()==at::kLong
        && tables.scalar_type()==at::kLong && lengths.scalar_type()==at::kLong, "paged prefill metadata must be int64");
    TORCH_CHECK(starts.dim()==1 && lengths.dim()==1 && tables.dim()==2 && offsets.dim()==1
        && tables.size(0)==starts.numel() && lengths.numel()==starts.numel()
        && offsets.numel()==starts.numel()+1, "paged prefill metadata shape mismatch");
    TORCH_CHECK(q.size(0)*q.size(1)<=2147483647, "too many paged prefill rows");
    if (validate) {
        auto oh=offsets.to(at::kCPU), sh=starts.to(at::kCPU), th=tables.to(at::kCPU), lh=lengths.to(at::kCPU);
        const auto* op=oh.data_ptr<int64_t>(); const auto* sp=sh.data_ptr<int64_t>();
        const auto* tp=th.data_ptr<int64_t>(); const auto* lp=lh.data_ptr<int64_t>();
        TORCH_CHECK(op[0]==0 && op[starts.numel()]==q.size(0), "paged prefill offsets must cover Q");
        for (int64_t b=0; b<starts.numel(); ++b) {
            TORCH_CHECK(op[b]>=0 && op[b+1]>=op[b] && op[b+1]<=q.size(0), "paged prefill offsets are not monotone");
            TORCH_CHECK(sp[b]>=0 && lp[b]>=sp[b] && lp[b]-sp[b]==op[b+1]-op[b]
                && lp[b]<=tables.size(1)*block_size, "paged prefill context/query range mismatch");
            const auto used=(lp[b]+block_size-1)/block_size;
            for (int64_t j=0; j<used; ++j)
                TORCH_CHECK(tp[b*tables.size(1)+j]>=0 && tp[b*tables.size(1)+j]<kc.size(0), "paged prefill physical page out of range");
        }
    }
    auto output=at::empty_like(q);
    if (q.size(0)==0) return output;
    const auto stream=c10::cuda::getCurrentCUDAStream(q.get_device()).stream();
    if (q.scalar_type()==at::kHalf) launch_paged_prefill<at::Half>(q,kc,vc,offsets,starts,tables,lengths,output,block_size,stream);
    else launch_paged_prefill<at::BFloat16>(q,kc,vc,offsets,starts,tables,lengths,output,block_size,stream);
    return output;
}
