#include "rmsnorm_bf16.cuh"
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
__global__ void k(const float*x,float*y,int64_t rows,int64_t cols){int64_t r=blockIdx.x;if(r>=rows)return;float ss=0;for(int64_t j=0;j<cols;j++){float v=x[r*cols+j];ss+=v*v;}float inv=rsqrtf(ss/cols+1e-5f);for(int64_t j=threadIdx.x;j<cols;j+=blockDim.x)y[r*cols+j]=x[r*cols+j]*inv;}
void rmsnorm_bf16(const float*x,float*y,int64_t rows,int64_t cols){if(!x||!y||rows<0||cols<=0)throw std::invalid_argument("rmsnorm_bf16");if(rows)k<<<rows,256>>>(x,y,rows,cols);ck("rmsnorm_bf16");}}
