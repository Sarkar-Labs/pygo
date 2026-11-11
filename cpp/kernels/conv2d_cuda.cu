// Im2col + GEMM Conv2d CUDA kernel
#include <cuda.h>
#include "pygo/cuda.h"
namespace pygo::kernels {

template<typename T>
__global__ void im2col_kernel(
    const T* input, T* col,
    int N,int C,int H,int W,int kH,int kW,int oH,int oW,
    int pH,int pW,int sH,int sW,int dH,int dW)
{
    int tid = blockIdx.x*blockDim.x + threadIdx.x;
    int total = N*C*kH*kW*oH*oW;
    if (tid >= total) return;
    int ow=tid%oW; tid/=oW; int oh=tid%oH; tid/=oH;
    int kw=tid%kW; tid/=kW; int kh=tid%kH; tid/=kH;
    int c=tid%C; int n=tid/C;
    int ih=oh*sH - pH + kh*dH;
    int iw=ow*sW - pW + kw*dW;
    int ci=(n*(C*kH*kW*oH*oW)) + ((c*kH*kW+kh*kW+kw)*(oH*oW)) + oh*oW+ow;
    col[ci] = (ih>=0&&ih<H&&iw>=0&&iw<W) ? input[n*C*H*W+c*H*W+ih*W+iw] : T(0);
}

void launch_im2col(const void* inp, void* col,
    int N,int C,int H,int W,int kH,int kW,int oH,int oW,
    int pH,int pW,int sH,int sW,int dH,int dW,DType dt,cudaStream_t s)
{
    int total=N*C*kH*kW*oH*oW, threads=256, blocks=(total+threads-1)/threads;
    if (dt==DType::Float32)
        im2col_kernel<float><<<blocks,threads,0,s>>>((float*)inp,(float*)col,N,C,H,W,kH,kW,oH,oW,pH,pW,sH,sW,dH,dW);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace pygo::kernels
