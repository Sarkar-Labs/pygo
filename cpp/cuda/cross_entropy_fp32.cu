#include "pygo/cuda.h"
#include "cross_entropy_fp32.cuh"
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
__global__ void k(const float*x,const int32_t*t,float*o,int64_t rows,int64_t C){int r=blockIdx.x;if(r>=rows)return;float m=-CUDART_INF_F,s=0;for(int j=0;j<C;j++)m=fmaxf(m,x[r*C+j]);for(int j=0;j<C;j++)s+=expf(x[r*C+j]-m);int y=t[r];o[r]=-(x[r*C+y]-m-logf(s));}
float cross_entropy_fp32(const float*x,const int32_t*t,int64_t rows,int64_t C){if(!x||!t||rows<=0||C<=0)throw std::invalid_argument("cross_entropy_fp32");float*d;cudaMalloc(&d,rows*sizeof(float));k<<<rows,1>>>(x,t,d,rows,C);ck("cross_entropy_fp32");std::vector<float>h(rows);cudaMemcpy(h.data(),d,rows*sizeof(float),cudaMemcpyDeviceToHost);cudaFree(d);double sum=0;for(float z:h)sum+=z;return (float)(sum/rows);}}
