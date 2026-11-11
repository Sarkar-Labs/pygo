// PyGo C++ Kernel: Matrix Multiplication
// Dispatches to: CUDA TensorCore / Metal MPS / AVX512 / NEON
#include "../include/pygo/kernels.h"
#include "../cuda/matmul_fp16.cuh"
#include "../cpu/cpu_tensor.h"
#include "../simd/avx512.h"

namespace pygo {
namespace kernels {

Tensor matmul(const Tensor& a, const Tensor& b) {
    if (a.device() == Device::CUDA) {
        return cuda::matmul_tensorcore(a, b);  // cuBLAS GEMM / CUTLASS
    } else if (a.device() == Device::METAL) {
        return metal::matmul_mps(a, b);        // Metal Performance Shaders
    } else if (cpu::has_avx512()) {
        return cpu::matmul_avx512(a, b);       // AVX-512 FMA kernel
    } else if (cpu::has_neon()) {
        return cpu::matmul_neon(a, b);         // ARM NEON dotprod
    } else {
        return cpu::matmul_naive(a, b);        // Fallback
    }
}

} // namespace kernels
} // namespace pygo
