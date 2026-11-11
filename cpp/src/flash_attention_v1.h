#pragma once
#ifndef PYGO_COMPLETE_SRC_FLASH_ATTENTION_V1_H_H
#define PYGO_COMPLETE_SRC_FLASH_ATTENTION_V1_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for src/flash_attention_v1.h.
inline double complete_flash_attention_v1(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_flash_attention_v1_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
