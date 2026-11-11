#pragma once
#ifndef PYGO_COMPLETE_TRAINING_LOSS_SCALE_H_H
#define PYGO_COMPLETE_TRAINING_LOSS_SCALE_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for training/loss_scale.h.
inline double complete_loss_scale(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_loss_scale_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
