#include "embedding_bf16.cuh"
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
__global__ void k(const __nv_bfloat16*w,const int64_t*idx,__nv_bfloat16*o,int64_t count,int64_t dim){for(int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;p<count*dim;p+=(int64_t)gridDim.x*blockDim.x)o[p]=w[idx[p/dim]*dim+p%dim];}
void embedding_bf16(const void*w,const int64_t*idx,void*out,int64_t count,int64_t dim){if(!w||!idx||!out||count<0||dim<=0)throw std::invalid_argument("embedding_bf16");int64_t n=count*dim;if(n)k<<<nb(n),256>>>((const __nv_bfloat16*)w,idx,(__nv_bfloat16*)out,count,dim);ck("embedding_bf16");}}
