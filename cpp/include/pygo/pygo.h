#pragma once
// PyGo — Main public header. Include this for the full API.
// Architecture: Go → Python → C++
// License: Apache 2.0

#define PYGO_VERSION_MAJOR 0
#define PYGO_VERSION_MINOR 1
#define PYGO_VERSION_PATCH 0
#define PYGO_VERSION "0.1.0"

// Core
#include "pygo/tensor.h"
#include "pygo/device.h"
#include "pygo/types.h"

// Neural Networks
#include "pygo/nn.h"

// Optimizers
#include "pygo/optim.h"

// Autograd
#include "pygo/autograd.h"

// Data
#include "pygo/data.h"

// Serialization
#include "pygo/serialization.h"

// Distributed
#include "pygo/distributed.h"

// CUDA (included if CUDA is available)
#ifdef PYGO_CUDA
#include "pygo/cuda.h"
#endif

// Metal (included if Metal is available)
#ifdef PYGO_METAL
#include "pygo/metal.h"
#endif

namespace pygo {

// ─── Version ─────────────────────────────────────────────────────────────────
struct Version {
    static int major() { return PYGO_VERSION_MAJOR; }
    static int minor() { return PYGO_VERSION_MINOR; }
    static int patch() { return PYGO_VERSION_PATCH; }
    static const char* str() { return PYGO_VERSION; }
};

// ─── Global Init ─────────────────────────────────────────────────────────────
struct InitOptions {
    bool enable_cuda    = true;
    bool enable_metal   = true;
    bool enable_opencl  = false;
    bool enable_vulkan  = false;
    int  num_threads    = -1;   // -1 = auto (use all cores)
    bool deterministic  = false;
    int  seed           = -1;   // -1 = random
    int  log_level      = 1;    // 0=silent 1=info 2=debug
};

// Initialise the PyGo runtime.
// Must be called once before any tensor operations.
void init(const InitOptions& opts = {});

// Clean up all resources.
void shutdown();

// Query runtime capabilities.
bool has_cuda();
bool has_metal();
bool has_opencl();
int  num_cuda_devices();

// Set global random seed.
void manual_seed(uint64_t seed);

// Synchronise all device streams.
void synchronize();

} // namespace pygo
