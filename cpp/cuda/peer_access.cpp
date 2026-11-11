 #include <cuda_runtime.h>
#include "peer_access.h"
namespace pygo::cuda {
bool can_access_peer(int dst,int src){int x=0;return cudaDeviceCanAccessPeer(&x,dst,src)==cudaSuccess&&x;} }
