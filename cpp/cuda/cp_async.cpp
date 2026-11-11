#include <cuda_runtime.h>
#include "cp_async.h"
namespace pygo::cuda { 
void cp_async_copy(void* dst,const void* src,size_t bytes){cudaMemcpyAsync(dst,src,bytes,cudaMemcpyDeviceToDevice);} }
