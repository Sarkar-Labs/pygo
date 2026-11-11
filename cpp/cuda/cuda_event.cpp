#include <cuda_runtime.h>
#include "cuda_event.h"
namespace pygo::cuda { 
void* event_create(){cudaEvent_t e=nullptr;cudaEventCreate(&e);return e;} void event_destroy(void* p){if(p)cudaEventDestroy((cudaEvent_t)p);} }
