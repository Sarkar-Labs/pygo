#include "sort_fp32.cuh"
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
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
void sort_fp32(const float*x,float*y,int64_t n){if(!x||!y||n<0)throw std::invalid_argument("sort_fp32");if(!n)return;cudaMemcpy(y,x,n*sizeof(float),cudaMemcpyDeviceToDevice);thrust::sort(thrust::device_pointer_cast(y),thrust::device_pointer_cast(y+n));ck("sort_fp32");}}
