 #include <cuda_runtime.h>
#include "cuda_driver.h"
namespace pygo::cuda {
int device_count(){int n=0;cudaGetDeviceCount(&n);return n;} int current_device(){int d=0;cudaGetDevice(&d);return d;} }
