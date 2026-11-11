#pragma once
#ifndef PYGO_COMPLETE_CPU_RW_LOCK_H_H
#define PYGO_COMPLETE_CPU_RW_LOCK_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for cpu/rw_lock.h.
inline double complete_rw_lock(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_rw_lock_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
