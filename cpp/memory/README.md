# PyGo Memory

## What this directory is

`cpp/memory` is a **native C++ component** of PyGo. It contains the implementation units, interfaces, tests, or build assets that belong to this subsystem. This README documents the real role of the directory and how it connects to the rest of the project; it is not a claim that every optional backend is enabled on every machine.

## How it works

PyGo is organized as layers. Public APIs describe tensors and higher-level operations, native implementations perform the work, and backend-specific code handles device or runtime details. Code in this directory should therefore be used through the public interface of its parent subsystem whenever one exists, rather than by depending on private implementation details.

The directory currently contains **46 files**. The file types present are: `.cpp` (22), `.h` (22), `.md` (1), `.txt` (1).

## Source inventory

The current directory includes:

- `CMakeLists.txt`
- `allocator.cpp`
- `allocator.h`
- `arena_allocator.cpp`
- `arena_allocator.h`
- `buddy_allocator.cpp`
- `buddy_allocator.h`
- `cache_allocator.cpp`
- `cache_allocator.h`
- `defragment.cpp`
- `defragment.h`
- `device_allocator.cpp`
- `device_allocator.h`
- `gc.cpp`
- `gc.h`
- `intrusive_ptr.cpp`
- `intrusive_ptr.h`
- `linear_allocator.cpp`
- `linear_allocator.h`
- `memory_arena.cpp`
- `memory_arena.h`
- `memory_block.cpp`
- `memory_block.h`
- `memory_fragment.cpp`
- `memory_fragment.h`
- `memory_pool.cpp`
- `memory_pool.h`
- `pinned_allocator.cpp`
- `pinned_allocator.h`
- `pool_allocator.cpp`
- `pool_allocator.h`
- `ref_count.cpp`
- `ref_count.h`
- `shared_memory.cpp`
- `shared_memory.h`
- `shared_ptr.cpp`
- `shared_ptr.h`
- `slab_allocator.cpp`
- `slab_allocator.h`
- `stack_allocator.cpp`
- Plus 5 additional files in this directory.

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
