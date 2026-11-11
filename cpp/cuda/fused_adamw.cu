#include "fused_adamw.cuh"
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
__global__ void k(float*p,const float*g,float*m,float*v,int64_t n,float lr,float b1,float b2,float eps,float wd,float step){for(int64_t i=(int64_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(int64_t)gridDim.x*blockDim.x){float z=g[i];m[i]=b1*m[i]+(1-b1)*z;v[i]=b2*v[i]+(1-b2)*z*z;float mh=m[i]/(1-powf(b1,step)),vh=v[i]/(1-powf(b2,step));p[i]-=lr*(mh/(sqrtf(vh)+eps)+wd*p[i]);}}
void fused_adamw(float*p,const float*g,float*m,float*v,int64_t n,float lr,float b1,float b2,float eps,float wd,float step,int64_t){if(!p||!g||!m||!v||n<0)throw std::invalid_argument("fused_adamw");if(n)k<<<nb(n),256>>>(p,g,m,v,n,lr,b1,b2,eps,wd,step);ck("fused_adamw");}}
