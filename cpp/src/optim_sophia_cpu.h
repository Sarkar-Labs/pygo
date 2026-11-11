#pragma once
#ifndef PYGO_COMPLETE_SRC_OPTIM_SOPHIA_CPU_H_H
#define PYGO_COMPLETE_SRC_OPTIM_SOPHIA_CPU_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for src/optim_sophia_cpu.h.
inline double complete_optim_sophia_cpu(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_optim_sophia_cpu_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
