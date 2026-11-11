#pragma once
#ifndef PYGO_COMPLETE_NN_MULTI_HEAD_ATTENTION_H_H
#define PYGO_COMPLETE_NN_MULTI_HEAD_ATTENTION_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for nn/multi_head_attention.h.
inline double complete_multi_head_attention(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_multi_head_attention_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
