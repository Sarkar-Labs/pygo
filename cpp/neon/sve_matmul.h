#pragma once
#ifndef PYGO_COMPLETE_NEON_SVE_MATMUL_H_H
#define PYGO_COMPLETE_NEON_SVE_MATMUL_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for neon/sve_matmul.h.
inline double complete_sve_matmul(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_sve_matmul_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
