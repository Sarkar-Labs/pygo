#include "fused_layernorm.cuh"
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
__global__ void k(const float*x,const float*g,const float*b,float*y,int64_t rows,int64_t C,float eps){int r=blockIdx.x;if(r>=rows)return;float m=0,v=0;for(int j=0;j<C;j++)m+=x[r*C+j];m/=C;for(int j=0;j<C;j++){float d=x[r*C+j]-m;v+=d*d;}float inv=rsqrtf(v/C+eps);for(int j=threadIdx.x;j<C;j+=blockDim.x)y[r*C+j]=(x[r*C+j]-m)*g[j]+b[j];}
void fused_layernorm(const float*x,const float*g,const float*b,float*y,int64_t rows,int64_t C,float eps){if(!x||!g||!b||!y||rows<0||C<=0)throw std::invalid_argument("fused_layernorm");if(rows)k<<<rows,256>>>(x,g,b,y,rows,C,eps);ck("fused_layernorm");}}
