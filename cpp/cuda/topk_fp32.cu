#include "pygo/cuda.h"
#include "topk_fp32.cuh"
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
__global__ void k(const float*x,float*v,int32_t*idx,int64_t rows,int K){int r=blockIdx.x;if(r>=rows)return;for(int j=0;j<K;j++){v[r*K+j]=-CUDART_INF_F;idx[r*K+j]=-1;}for(int j=0;j<rows;j++){float z=x[r*rows+j];int pos=K;for(int q=0;q<K;q++)if(z>v[r*K+q]){pos=q;break;}if(pos<K){for(int q=K-1;q>pos;q--){v[r*K+q]=v[r*K+q-1];idx[r*K+q]=idx[r*K+q-1];}v[r*K+pos]=z;idx[r*K+pos]=j;}}}
void topk_fp32(const float*x,float*v,int32_t*idx,int64_t n,int32_t K){if(!x||!v||!idx||n<=0||K<=0||K>n)throw std::invalid_argument("topk_fp32");k<<<n,1>>>(x,v,idx,n,K);ck("topk_fp32");}}
