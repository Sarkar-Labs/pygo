#pragma once
#ifndef PYGO_COMPLETE_RUNTIME_PARALLEL_FOR_H_H
#define PYGO_COMPLETE_RUNTIME_PARALLEL_FOR_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for runtime/parallel_for.h.
inline double complete_parallel_for(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_parallel_for_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
