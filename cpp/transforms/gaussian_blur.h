#pragma once
#ifndef PYGO_COMPLETE_TRANSFORMS_GAUSSIAN_BLUR_H_H
#define PYGO_COMPLETE_TRANSFORMS_GAUSSIAN_BLUR_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for transforms/gaussian_blur.h.
inline double complete_gaussian_blur(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_gaussian_blur_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
