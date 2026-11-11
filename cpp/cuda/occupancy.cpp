 #include <cuda_runtime.h>
#include "occupancy.h"
namespace pygo::cuda {
int max_active_blocks(const void* kernel,int threads){int n=0;cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n,(const void*)kernel,threads,0);return n;} }
