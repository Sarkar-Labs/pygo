#pragma once
#ifndef PYGO_COMPLETE_INCLUDE_PYGO_CUDA_H_H
#define PYGO_COMPLETE_INCLUDE_PYGO_CUDA_H_H

#include "pygo/tensor.h"
#include <cstddef>
#include <stdexcept>
#include <string>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <math_constants.h>

namespace pygo::cuda {

inline void check_cuda(cudaError_t status, const char* expression, const char* file, int line) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string("PyGo CUDA error at ") + file + ":" +
                                 std::to_string(line) + " in " + expression + ": " +
                                 cudaGetErrorString(status));
    }
}

#define PYGO_CUDA_CHECK(expr) ::pygo::cuda::check_cuda((expr), #expr, __FILE__, __LINE__)
// Backward-compatible alias used by older CUDA translation units.
#ifndef CUDA_CHECK
#define CUDA_CHECK(expr) PYGO_CUDA_CHECK(expr)
#endif
#ifndef CUDART_INF_F
#define CUDART_INF_F __int_as_float(0x7f800000)
#endif

inline bool is_available() noexcept {
    int count = 0;
    cudaError_t status = cudaGetDeviceCount(&count);
    if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver) {
        cudaGetLastError();
        return false;
    }
    if (status != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    return count > 0;
}

inline void* to_device(const void* host, std::size_t bytes) {
    if (bytes == 0) return nullptr;
    if (!host) throw std::invalid_argument("PyGo CUDA: null host pointer");
    void* device = nullptr;
    PYGO_CUDA_CHECK(cudaMalloc(&device, bytes));
    try {
        PYGO_CUDA_CHECK(cudaMemcpy(device, host, bytes, cudaMemcpyHostToDevice));
    } catch (...) {
        cudaFree(device);
        throw;
    }
    return device;
}

inline void to_host(void* host, const void* device, std::size_t bytes) {
    if (bytes == 0) return;
    if (!host || !device) throw std::invalid_argument("PyGo CUDA: null copy pointer");
    PYGO_CUDA_CHECK(cudaMemcpy(host, device, bytes, cudaMemcpyDeviceToHost));
}

inline void free_device(void* device) noexcept {
    if (device) cudaFree(device);
}

} // namespace pygo::cuda
#endif // __CUDACC__

namespace pygo::component {
// CPU-safe declarations used by the public header when CUDA is not enabled.
inline double complete_cuda(const double* x, std::size_t n) {
    if (!x || n == 0) return 0.0;
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) sum += x[i];
    return sum / static_cast<double>(n);
}

inline void complete_cuda_copy(const double* x, double* y, std::size_t n) {
    if (!x || !y) return;
    for (std::size_t i = 0; i < n; ++i) y[i] = x[i];
}
} // namespace pygo::component

#endif
