#pragma once
#ifndef PYGO_COMPLETE_CORE_DEAD_CODE_ELIM_H_H
#define PYGO_COMPLETE_CORE_DEAD_CODE_ELIM_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for core/dead_code_elim.h.
inline double complete_dead_code_elim(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_dead_code_elim_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
