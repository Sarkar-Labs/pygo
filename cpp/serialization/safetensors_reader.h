#pragma once
#ifndef PYGO_COMPLETE_SERIALIZATION_SAFETENSORS_READER_H_H
#define PYGO_COMPLETE_SERIALIZATION_SAFETENSORS_READER_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for serialization/safetensors_reader.h.
inline double complete_safetensors_reader(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_safetensors_reader_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
