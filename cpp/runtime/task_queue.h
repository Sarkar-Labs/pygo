#pragma once
#ifndef PYGO_COMPLETE_RUNTIME_TASK_QUEUE_H_H
#define PYGO_COMPLETE_RUNTIME_TASK_QUEUE_H_H
#include "generated_runtime.h"

namespace pygo::component {
// Stable, dependency-light implementation surface for runtime/task_queue.h.
inline double complete_task_queue(const double* x, std::size_t n) { return x && n ? pygo::runtime::mean(x,n) : 0.0; }
inline void complete_task_queue_copy(const double* x, double* y, std::size_t n) { if(x && y) pygo::runtime::copy(x,y,n); }
}
#endif
