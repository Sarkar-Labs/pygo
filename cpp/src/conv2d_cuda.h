#pragma once
#ifndef PYGO_COMPLETE_SRC_CONV2D_CUDA_H_H
#define PYGO_COMPLETE_SRC_CONV2D_CUDA_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for src/conv2d_cuda.h.
inline double complete_conv2d_cuda(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_conv2d_cuda_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
