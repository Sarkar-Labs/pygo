#pragma once
#ifndef PYGO_COMPLETE_LOSSES_CROSS_ENTROPY_H_H
#define PYGO_COMPLETE_LOSSES_CROSS_ENTROPY_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for losses/cross_entropy.h.
inline double complete_cross_entropy(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_cross_entropy_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
