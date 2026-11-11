# CUDA Diagnostic Fix

This release addresses the three CUDA compilation blockers reported by the external 80-file `nvcc` diagnostic.

## 1. `Tensor::mutable_data<T>()`

`pygo/cpp/include/pygo/tensor.h` now provides mutable and const overloads of `mutable_data<T>()`. The accessor returns the existing tensor storage pointer and does not allocate or copy.

## 2. `CUDART_INF_F`

`pygo/cpp/include/pygo/cuda.h` now includes CUDA math constants and provides a compatibility fallback for toolkits that do not expose `CUDART_INF_F` through the included headers.

CUDA translation units using the infinity constant explicitly include `pygo/cuda.h`.

## 3. `CUDA_CHECK`

`pygo/cpp/include/pygo/cuda.h` now provides `CUDA_CHECK(expr)` as a compatibility alias to PyGo's checked CUDA error wrapper, `PYGO_CUDA_CHECK(expr)`.

## Validation

- 80 `.cu` translation units present.
- No helper-header dependency gaps for `CUDART_INF_F`, `CUDA_CHECK`, or `PYGO_CUDA_CHECK`.
- `tensor.cpp` C++17 syntax validation: PASS.
- CPU CMake configuration: PASS.
- Full CUDA `nvcc` compilation cannot be executed in this environment because NVIDIA CUDA Toolkit / `nvcc` is not installed. The repository's CUDA validation target is configured to compile every `.cu` file on a CUDA-enabled machine.
