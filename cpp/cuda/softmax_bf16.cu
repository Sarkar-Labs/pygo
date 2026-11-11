#include "softmax_bf16.cuh"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>
namespace pygo::cuda {

namespace {
__global__ void k(const float*x,float*y,int64_t rows,int64_t cols){int r=blockIdx.x;if(r>=rows)return;float m=-INFINITY;for(int j=threadIdx.x;j<cols;j+=blockDim.x)m=fmaxf(m,x[r*cols+j]);__shared__ float sm[256];sm[threadIdx.x]=m;__syncthreads();for(int d=128;d;d>>=1){if(threadIdx.x<d)sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+d]);__syncthreads();}m=sm[0];float sum=0;for(int j=threadIdx.x;j<cols;j+=blockDim.x)sum+=expf(x[r*cols+j]-m);sm[threadIdx.x]=sum;__syncthreads();for(int d=128;d;d>>=1){if(threadIdx.x<d)sm[threadIdx.x]+=sm[threadIdx.x+d];__syncthreads();}sum=sm[0];for(int j=threadIdx.x;j<cols;j+=blockDim.x)y[r*cols+j]=expf(x[r*cols+j]-m)/sum;}
}

void softmax_bf16(const float*x,float*y,int64_t rows,int64_t cols){if (!x || !y || rows < 0 || cols <= 0) throw std::invalid_argument("softmax_bf16: invalid shape");
    if (rows == 0) return;
    k<<<static_cast<unsigned>(rows), 256>>>(x, y, rows, cols);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

} // namespace pygo::cuda
