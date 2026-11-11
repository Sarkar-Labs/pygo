#include "scan_fp32.cuh"
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
__global__ void k(const float*a,float*b,int64_t n){if(threadIdx.x||blockIdx.x)return;float acc=0;for(int64_t i=0;i<n;i++){acc+=a[i];b[i]=acc;}}
void scan_fp32(const float*a,float*b,int64_t n){if(!a||!b||n<0)throw std::invalid_argument("scan_fp32");if(n)k<<<1,1>>>(a,b,n);ck("scan_fp32");}}
