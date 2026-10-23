// Stable greedy argmax: ties and NaNs select the lowest vocabulary index,
// matching torch.argmax. No host synchronization; capture-safe scratch tensors.
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <climits>

namespace {
constexpr int THREADS=256, CHUNK=4096;
__device__ bool better(float a, int ai, float b, int bi) {
    if (isnan(a)) return !isnan(b) || ai<bi;
    if (isnan(b)) return false;
    return a>b || (a==b && ai<bi);
}
__device__ void reduce(float& v, int& index, float* values, int* indices) {
    int tid=threadIdx.x;
    values[tid]=v; indices[tid]=index;
    __syncthreads();
    for (int stride=THREADS/2; stride; stride/=2) {
        if (tid<stride && better(values[tid+stride],indices[tid+stride],values[tid],indices[tid])) {
            values[tid]=values[tid+stride]; indices[tid]=indices[tid+stride];
        }
        __syncthreads();
    }
    v=values[0]; index=indices[0];
}
template <typename T>
__global__ void partial_argmax(const T* x, float* values, int* indices, int vocab, int parts) {
    __shared__ float sv[THREADS]; __shared__ int si[THREADS];
    float best=-INFINITY; int index=INT_MAX;
    int begin=blockIdx.x*CHUNK, end=min(begin+CHUNK,vocab);
    for (int i=begin+threadIdx.x;i<end;i+=THREADS) {
        float v=static_cast<float>(x[static_cast<int64_t>(blockIdx.y)*vocab+i]);
        if (better(v,i,best,index)) { best=v; index=i; }
    }
    reduce(best,index,sv,si);
    if (threadIdx.x==0) { values[blockIdx.y*parts+blockIdx.x]=best; indices[blockIdx.y*parts+blockIdx.x]=index; }
}
__global__ void final_argmax(const float* values,const int* indices,int64_t* out,int parts) {
    __shared__ float sv[THREADS]; __shared__ int si[THREADS];
    float best=-INFINITY; int index=INT_MAX;
    for (int i=threadIdx.x;i<parts;i+=THREADS) {
        float v=values[blockIdx.x*parts+i]; int j=indices[blockIdx.x*parts+i];
        if (better(v,j,best,index)) { best=v; index=j; }
    }
    reduce(best,index,sv,si);
    if (threadIdx.x==0) out[blockIdx.x]=index;
}
}
at::Tensor greedy_argmax_cuda(const at::Tensor& logits) {
    const c10::cuda::CUDAGuard guard(logits.device());
    auto out=at::empty({logits.size(0)},logits.options().dtype(at::kLong));
    if (!logits.size(0)) return out;
    const int parts=(logits.size(1)+CHUNK-1)/CHUNK;
    auto values=at::empty({logits.size(0),parts},logits.options().dtype(at::kFloat));
    auto indices=at::empty({logits.size(0),parts},logits.options().dtype(at::kInt));
    auto stream=c10::cuda::getCurrentCUDAStream(logits.get_device()).stream();
    AT_DISPATCH_FLOATING_TYPES_AND2(at::kHalf,at::kBFloat16,logits.scalar_type(),"greedy_argmax",[&] {
        partial_argmax<scalar_t><<<dim3(parts,logits.size(0)),THREADS,0,stream>>>(
            logits.data_ptr<scalar_t>(),values.data_ptr<float>(),indices.data_ptr<int>(),logits.size(1),parts);
    });
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    final_argmax<<<logits.size(0),THREADS,0,stream>>>(values.data_ptr<float>(),indices.data_ptr<int>(),out.data_ptr<int64_t>(),parts);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return out;
}
