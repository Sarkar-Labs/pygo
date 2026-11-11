 #include <cuda_runtime.h>
#include "cuda_unified.h"
namespace pygo::cuda {
void* unified_alloc(size_t bytes){void*p=nullptr;cudaMallocManaged(&p,bytes);return p;} void unified_free(void*p){if(p)cudaFree(p);} }
