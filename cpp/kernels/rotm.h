#pragma once
#ifndef PYGO_COMPLETE_KERNELS_ROTM_H_H
#define PYGO_COMPLETE_KERNELS_ROTM_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for kernels/rotm.h.
inline double complete_rotm(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_rotm_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
