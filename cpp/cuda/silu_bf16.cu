#include "silu_bf16.cuh"
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
__global__ void k(const float*a,float*b,int64_t n){for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x){float x=a[i];b[i]=x/(1.f+expf(-x));}}
void silu_bf16(const float*a,float*b,int64_t rows,int64_t cols){if(!a||!b||rows<0||cols<0)throw std::invalid_argument("silu_bf16");int64_t n=rows*cols;if(n)k<<<nb(n),256>>>(a,b,n);ck("silu_bf16");}}
