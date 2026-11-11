# PyGo Core

## What this directory is

`cpp/core` is a **native C++ component** of PyGo. It contains the implementation units, interfaces, tests, or build assets that belong to this subsystem. This README documents the real role of the directory and how it connects to the rest of the project; it is not a claim that every optional backend is enabled on every machine.

## How it works

PyGo is organized as layers. Public APIs describe tensors and higher-level operations, native implementations perform the work, and backend-specific code handles device or runtime details. Code in this directory should therefore be used through the public interface of its parent subsystem whenever one exists, rather than by depending on private implementation details.

The directory currently contains **51 files**. The file types present are: `.cpp` (25), `.h` (24), `.md` (1), `.txt` (1).

## Source inventory

The current directory includes:

- `CMakeLists.txt`
- `autocast.cpp`
- `autocast.h`
- `backend_select.cpp`
- `backend_select.h`
- `constant_folding.cpp`
- `constant_folding.h`
- `context.cpp`
- `context.h`
- `dead_code_elim.cpp`
- `dead_code_elim.h`
- `device_guard.cpp`
- `device_guard.h`
- `dispatch.cpp`
- `dispatch.h`
- `dtype_meta.cpp`
- `dtype_meta.h`
- `execution_plan.cpp`
- `execution_plan.h`
- `fusion_pass.cpp`
- `fusion_pass.h`
- `ir.cpp`
- `ir_graph.cpp`
- `ir_graph.h`
- `ir_node.cpp`
- `ir_node.h`
- `kernel_registry.cpp`
- `kernel_registry.h`
- `memory_planning.cpp`
- `memory_planning.h`
- `operator_registry.cpp`
- `operator_registry.h`
- `optimization_pass.cpp`
- `optimization_pass.h`
- `pass_manager.cpp`
- `pass_manager.h`
- `schema.cpp`
- `schema.h`
- `scope.cpp`
- `scope.h`
- Plus 10 additional files in this directory.

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
