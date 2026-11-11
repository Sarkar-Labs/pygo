#pragma once
#ifndef PYGO_COMPLETE_NN_CONV_TRANSPOSE1D_H_H
#define PYGO_COMPLETE_NN_CONV_TRANSPOSE1D_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for nn/conv_transpose1d.h.
inline double complete_conv_transpose1d(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_conv_transpose1d_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
