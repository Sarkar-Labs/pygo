#include "pygo/cuda.h"
#include "reduce_max_fp16.cuh"
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>
#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>
#include <algorithm>
namespace pygo::cuda {
namespace { inline void ck(const char*n){auto e=cudaGetLastError();if(e!=cudaSuccess)throw std::runtime_error(std::string(n)+": "+cudaGetErrorString(e));} inline int nb(int64_t n){return (int)((n+255)/256);} }
__global__ void k(const float*a,float*out,int64_t n){__shared__ float sm[256];float v=-CUDART_INF_F;for(int64_t i=threadIdx.x;i<n;i+=blockDim.x)v=fmaxf(v,a[i]);sm[threadIdx.x]=v;__syncthreads();for(int d=128;d;d>>=1){if(threadIdx.x<d)sm[threadIdx.x]=fmaxf(sm[threadIdx.x],sm[threadIdx.x+d]);__syncthreads();}if(threadIdx.x==0)*out=sm[0];}
void reduce_max_fp16(const float*a,float*out,int64_t n){if(!a||!out||n<=0)throw std::invalid_argument("reduce_max_fp16");k<<<1,256>>>(a,out,n);ck("reduce_max_fp16");}}
