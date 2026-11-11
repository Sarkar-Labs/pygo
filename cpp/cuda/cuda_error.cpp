#include <cuda_runtime.h>
#include "cuda_error.h"
namespace pygo::cuda { 
const char* cuda_error_string(int code){return cudaGetErrorString((cudaError_t)code);} }
