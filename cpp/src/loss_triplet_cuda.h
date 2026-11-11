#pragma once
#ifndef PYGO_COMPLETE_SRC_LOSS_TRIPLET_CUDA_H_H
#define PYGO_COMPLETE_SRC_LOSS_TRIPLET_CUDA_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for src/loss_triplet_cuda.h.
inline double complete_loss_triplet_cuda(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_loss_triplet_cuda_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
