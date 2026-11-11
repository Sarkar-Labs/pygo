#include "rope_fp16.cuh"
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
__global__ void k(float*x,int64_t rows,int64_t dim,float base){int64_t r=blockIdx.x;if(r>=rows)return;for(int64_t j=threadIdx.x;j<dim/2;j+=blockDim.x){int64_t i=2*j;float a=x[r*dim+i],b=x[r*dim+i+1],th=r*powf(base,-(float)i/dim),c=cosf(th),sn=sinf(th);x[r*dim+i]=a*c-b*sn;x[r*dim+i+1]=a*sn+b*c;}}
void rope_fp16(float*x,int64_t rows,int64_t dim,float base){if(!x||rows<0||dim<=0||dim%2)throw std::invalid_argument("rope_fp16");if(rows)k<<<rows,256>>>(x,rows,dim,base);ck("rope_fp16");}}
