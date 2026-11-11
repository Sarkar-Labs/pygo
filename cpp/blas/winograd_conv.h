#pragma once
#ifndef PYGO_COMPLETE_BLAS_WINOGRAD_CONV_H_H
#define PYGO_COMPLETE_BLAS_WINOGRAD_CONV_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for blas/winograd_conv.h.
inline double complete_winograd_conv(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_winograd_conv_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
