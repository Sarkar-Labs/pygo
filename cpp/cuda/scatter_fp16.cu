#include "scatter_fp16.cuh"
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
__global__ void k(const float*src,const int64_t*idx,float*dst,int64_t count,int64_t dim){for(int64_t p=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;p<count*dim;p+=(int64_t)gridDim.x*blockDim.x){int64_t r=p/dim,c=p%dim;dst[idx[r]*dim+c]=src[p];}}
void scatter_fp16(const float*src,const int64_t*idx,float*dst,int64_t count,int64_t dim){if(!src||!idx||!dst||count<0||dim<=0)throw std::invalid_argument("scatter_fp16");int64_t n=count*dim;if(n)k<<<nb(n),256>>>(src,idx,dst,count,dim);ck("scatter_fp16");}}
