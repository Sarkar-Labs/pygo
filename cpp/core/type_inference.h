#pragma once
#ifndef PYGO_COMPLETE_CORE_TYPE_INFERENCE_H_H
#define PYGO_COMPLETE_CORE_TYPE_INFERENCE_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for core/type_inference.h.
inline double complete_type_inference(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_type_inference_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
