#include "matmul_bf16.cuh"
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
__global__ void k(const float*A,const float*B,float*C,int M,int N,int K,float alpha,float beta){int r=blockIdx.y*blockDim.y+threadIdx.y,c=blockIdx.x*blockDim.x+threadIdx.x;if(r<M&&c<N){float s=0;for(int k=0;k<K;k++)s+=A[r*K+k]*B[k*N+c];C[r*N+c]=alpha*s+beta*C[r*N+c];}}
void matmul_bf16(const float*A,const float*B,float*C,int M,int N,int K,float alpha,float beta){if(!A||!B||!C||M<0||N<0||K<0)throw std::invalid_argument("matmul_bf16");dim3 t(16,16),g((N+15)/16,(M+15)/16);k<<<g,t>>>(A,B,C,M,N,K,alpha,beta);ck("matmul_bf16");}}
