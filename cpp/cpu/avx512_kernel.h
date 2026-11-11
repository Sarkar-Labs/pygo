#pragma once
#ifndef PYGO_COMPLETE_CPU_AVX512_KERNEL_H_H
#define PYGO_COMPLETE_CPU_AVX512_KERNEL_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for cpu/avx512_kernel.h.
inline double complete_avx512_kernel(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_avx512_kernel_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
