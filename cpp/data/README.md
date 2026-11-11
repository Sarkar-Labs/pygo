# PyGo Data

## What this directory is

`cpp/data` is a **native C++ component** of PyGo. It contains the implementation units, interfaces, tests, or build assets that belong to this subsystem. This README documents the real role of the directory and how it connects to the rest of the project; it is not a claim that every optional backend is enabled on every machine.

## How it works

PyGo is organized as layers. Public APIs describe tensors and higher-level operations, native implementations perform the work, and backend-specific code handles device or runtime details. Code in this directory should therefore be used through the public interface of its parent subsystem whenever one exists, rather than by depending on private implementation details.

The directory currently contains **34 files**. The file types present are: `.cpp` (16), `.h` (16), `.md` (1), `.txt` (1).

## Source inventory

The current directory includes:

- `CMakeLists.txt`
- `arrow_reader.cpp`
- `arrow_reader.h`
- `audio_reader.cpp`
- `audio_reader.h`
- `binary_reader.cpp`
- `binary_reader.h`
- `cache.cpp`
- `cache.h`
- `collate.cpp`
- `collate.h`
- `csv_reader.cpp`
- `csv_reader.h`
- `dataloader.cpp`
- `dataloader.h`
- `dataset.cpp`
- `dataset.h`
- `hdf5_reader.cpp`
- `hdf5_reader.h`
- `image_reader.cpp`
- `image_reader.h`
- `parquet_reader.cpp`
- `parquet_reader.h`
- `prefetch.cpp`
- `prefetch.h`
- `sampler.cpp`
- `sampler.h`
- `text_reader.cpp`
- `text_reader.h`
- `video_reader.cpp`
- `video_reader.h`
- `zarr_reader.cpp`
- `zarr_reader.h`

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
