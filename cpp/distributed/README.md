# PyGo Distributed

## What this directory is

`cpp/distributed` is a **native C++ component** of PyGo. It contains the implementation units, interfaces, tests, or build assets that belong to this subsystem. This README documents the real role of the directory and how it connects to the rest of the project; it is not a claim that every optional backend is enabled on every machine.

## How it works

PyGo is organized as layers. Public APIs describe tensors and higher-level operations, native implementations perform the work, and backend-specific code handles device or runtime details. Code in this directory should therefore be used through the public interface of its parent subsystem whenever one exists, rather than by depending on private implementation details.

The directory currently contains **58 files**. The file types present are: `.cpp` (28), `.h` (28), `.md` (1), `.txt` (1).

## Source inventory

The current directory includes:

- `CMakeLists.txt`
- `all_gather.cpp`
- `all_gather.h`
- `all_reduce.cpp`
- `all_reduce.h`
- `backend.cpp`
- `backend.h`
- `barrier.cpp`
- `barrier.h`
- `broadcast.cpp`
- `broadcast.h`
- `comm.cpp`
- `comm.h`
- `data_parallel.cpp`
- `data_parallel.h`
- `fsdp.cpp`
- `fsdp.h`
- `gloo_backend.cpp`
- `gloo_backend.h`
- `gradient_compression.cpp`
- `gradient_compression.h`
- `model_parallel.cpp`
- `model_parallel.h`
- `mpi_backend.cpp`
- `mpi_backend.h`
- `nccl_backend.cpp`
- `nccl_backend.h`
- `p2p.cpp`
- `p2p.h`
- `pipeline.cpp`
- `pipeline.h`
- `recursive_halving.cpp`
- `recursive_halving.h`
- `recv.cpp`
- `recv.h`
- `reduce.cpp`
- `reduce.h`
- `ring_allreduce.cpp`
- `ring_allreduce.h`
- `rpc.cpp`
- Plus 17 additional files in this directory.

## Build and integration

For a normal CPU development build, configure the repository from its root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPYGO_ENABLE_CUDA=AUTO
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

If this directory is optional, CMake only compiles it when its corresponding feature and toolchain are available. A successful CPU build must not be interpreted as proof that an accelerator implementation has been compiled.

## Using the component

Prefer the stable PyGo public API exposed by the parent package. Keep ownership, dtype, shape, device, and error handling at the public boundary. Do not pass borrowed raw pointers across a subsystem boundary unless the API explicitly documents their lifetime.

When this directory contains kernels or low-level helpers, callers are expected to validate inputs before dispatch and to propagate backend errors rather than silently falling back to an unrelated implementation.

## Testing

Tests for this component should cover:

1. normal valid inputs;
2. empty and boundary inputs;
3. invalid shapes, dtypes, and devices;
4. ownership and lifetime rules where buffers are involved;
5. numerical correctness against a trusted reference; and
6. backend-specific behavior when the required compiler or hardware is available.

For repository-wide verification, run `ctest --test-dir build --output-on-failure` and the relevant Python or Go test command from the repository root.

## Production notes

This directory is part of the PyGo source distribution. Production status is determined by the tests and build configuration for the specific feature, not by the existence of a source file or device enum. Unsupported optional features should return a clear error instead of producing incorrect results.

## Documentation rule

Keep this README synchronized with the actual files and build behavior. If an API, backend, or file is removed, update the inventory and usage instructions in the same change.
