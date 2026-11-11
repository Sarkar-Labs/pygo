#pragma once
#ifndef PYGO_COMPLETE_SCHEDULERS_STEP_LR_H_H
#define PYGO_COMPLETE_SCHEDULERS_STEP_LR_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for schedulers/step_lr.h.
inline double complete_step_lr(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_step_lr_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
