#include <cuda_runtime.h>
#include "cuda_allocator.h"
namespace pygo::cuda { 
void* cuda_alloc(size_t bytes){void*p=nullptr;if(cudaMalloc(&p,bytes)!=cudaSuccess)return nullptr;return p;} void cuda_free(void*p){if(p)cudaFree(p);} }
