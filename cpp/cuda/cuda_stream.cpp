#include <cuda_runtime.h>
#include "cuda_stream.h"
namespace pygo::cuda { 
void* stream_create(){cudaStream_t s=nullptr;cudaStreamCreate(&s);return s;} void stream_destroy(void*p){if(p)cudaStreamDestroy((cudaStream_t)p);} }
