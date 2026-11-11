#pragma once
#ifndef PYGO_COMPLETE_AVX_AVX512_ATTENTION_H_H
#define PYGO_COMPLETE_AVX_AVX512_ATTENTION_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for avx/avx512_attention.h.
inline double complete_avx512_attention(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_avx512_attention_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
