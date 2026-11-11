#include "lion_fp32.cuh"
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
__global__ void k(const float*g,float*p,int64_t n){for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x){float z=g[i];p[i]-=.001f*((z>0)-(z<0));}}
void lion_fp32(const float*g,float*p,int64_t n){if(!g||!p||n<0)throw std::invalid_argument("lion_fp32");if(n)k<<<nb(n),256>>>(g,p,n);ck("lion_fp32");}}
