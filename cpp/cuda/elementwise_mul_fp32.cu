#include "elementwise_mul_fp32.cuh"
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
__global__ void k(const float*a,const float*b,float*c,int64_t n){for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x)c[i]=a[i]*b[i];}
void elementwise_mul_fp32(const float*a,const float*b,float*c,int64_t n){if(!a||!b||!c||n<0)throw std::invalid_argument("elementwise_mul_fp32");if(n)k<<<nb(n),256>>>(a,b,c,n);ck("elementwise_mul_fp32");}}
