# PyGo CUDA Backend

## What this directory is

`cpp/cuda` contains PyGo's NVIDIA CUDA backend. It is the device-specific implementation layer used when PyGo is built with the CUDA toolkit and an NVIDIA GPU is available. The directory currently contains **80 `.cu` translation units**, **78 `.cuh` CUDA headers**, and **49 C++ support files**.

The CUDA implementation is separate from the CPU runtime so a CPU-only installation does not need the CUDA compiler. CUDA is enabled through the top-level CMake option `PYGO_ENABLE_CUDA`.

## How a CUDA operation works

The normal execution path is:

```text
PyGo Tensor/API
      |
      v
operation dispatch
      |
      v
CUDA wrapper / validation
      |
      v
CUDA kernel (`__global__`)
      |
      v
NVIDIA GPU memory
```

A CUDA translation unit normally validates pointers and dimensions, launches one or more kernels, checks the CUDA error state, and returns control to the tensor/runtime layer. Kernels use CUDA execution primitives such as `blockIdx`, `threadIdx`, shared memory, and CUDA runtime APIs where appropriate.

## What is actually implemented here

The `.cu` files contain real CUDA code rather than CPU-only placeholders. Examples include:

- attention kernels in FP32 and BF16;
- convolution kernels in FP16;
- AdamW update kernels;
- elementwise FP16 operations;
- softmax and reduction kernels;
- GELU, SiLU, RMSNorm, RoPE and scatter kernels;
- cumulative operations and sorting helpers;
- other dtype-specific CUDA operators.

A quick inspection of the source can be performed with:

```bash
grep -RInE '__global__|cudaMalloc|cudaMemcpy|cudaStream|cudaDevice|threadIdx|blockIdx' cpp/cuda --include='*.cu'
```

## Building with `nvcc`

PyGo's CUDA CMake project declares CUDA explicitly:

```cmake
project(pygo_cuda LANGUAGES CXX CUDA)
set(CMAKE_CUDA_STANDARD 17)
```

On a machine with the NVIDIA CUDA toolkit installed, verify the compiler first:

```bash
nvcc --version
```

Then configure PyGo:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYGO_ENABLE_CUDA=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

`PYGO_ENABLE_CUDA=ON` intentionally fails during configuration when CMake cannot find a CUDA compiler. This prevents a user from thinking a CPU-only build is a CUDA build.

For portable development, use:

```bash
cmake -S . -B build -DPYGO_ENABLE_CUDA=AUTO
```

`AUTO` uses CUDA when the toolchain is discoverable and otherwise keeps the CPU build usable.

## Compiler requirements

The CUDA sources require a compatible NVIDIA CUDA toolkit, host C++ compiler, and GPU driver. The exact supported GPU architectures should be selected by the release build configuration for the target deployment. Do not hard-code an architecture that is unavailable on the target GPU.

## Runtime behavior

CUDA kernels should:

1. validate dimensions and pointers;
2. launch with a valid grid and block configuration;
3. check launch errors with `cudaGetLastError()` or an equivalent mechanism;
4. synchronize only when required by the API contract;
5. propagate CUDA errors instead of swallowing them; and
6. preserve dtype and layout semantics expected by the tensor runtime.

Asynchronous execution is important for performance. A kernel launch does not necessarily mean that GPU work has completed, so callers must use the appropriate CUDA stream/event synchronization rule when reading results or releasing dependent resources.

## Memory and transfers

Device tensors should keep their storage on the GPU between operations. Host-to-device and device-to-host transfers should be explicit at the tensor/runtime boundary. When a zero-copy mechanism is available, it must still obey the documented ownership and lifetime rules; a raw pointer alone is not an ownership contract.

## Current verification status

The current development environment does **not** contain `nvcc` or an NVIDIA GPU, so CUDA hardware execution cannot honestly be certified here. The CUDA source has been inspected and the build system is configured to detect the compiler correctly. A CUDA-enabled release should run the CUDA test suite on a real NVIDIA CI runner before publishing hardware-specific performance claims.

## Debugging CUDA builds

Useful commands on a CUDA machine:

```bash
which nvcc
nvcc --version
cmake -S . -B build-cuda -DPYGO_ENABLE_CUDA=ON -DCMAKE_BUILD_TYPE=Debug
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

For runtime debugging, NVIDIA tools such as Compute Sanitizer can be used against the resulting test executable. Race conditions, out-of-bounds accesses, and synchronization mistakes should be treated as release-blocking defects.

## Production boundary

The presence of CUDA source files means the backend has source-level implementations; it does not mean every operator has been validated on every NVIDIA GPU. Production certification requires compilation with the intended CUDA toolkit, execution on representative GPUs, numerical comparison against CPU/reference implementations, memory-safety checks, and performance benchmarks.
