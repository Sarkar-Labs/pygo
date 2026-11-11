#pragma once
#ifndef PYGO_COMPLETE_SRC_FLASH_ATTENTION_V3_H_H
#define PYGO_COMPLETE_SRC_FLASH_ATTENTION_V3_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for src/flash_attention_v3.h.
inline double complete_flash_attention_v3(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_flash_attention_v3_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
