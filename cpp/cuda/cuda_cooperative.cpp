 #include <cuda_runtime.h>
#include "cuda_cooperative.h"
namespace pygo::cuda {
bool cooperative_launch_supported(){cudaDeviceProp p{};int d=0;if(cudaGetDevice(&d)!=cudaSuccess)return false;cudaGetDeviceProperties(&p,d);return p.cooperativeLaunch;} }
