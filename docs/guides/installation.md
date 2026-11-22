# Installation

## Requirements

For the CPU development path:

- CMake 3.20 or newer
- C++17 compiler
- Python 3.13 or newer
- Go 1.22 or newer
- NumPy 1.24 or newer for the Python package

CUDA builds additionally require a compatible NVIDIA CUDA toolkit and compiler.

## CPU build

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPYGO_ENABLE_CUDA=OFF -DPYGO_BUILD_TESTS=ON
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

## Python package

```bash
python -m pip install -e .
python -m pytest tests/python -q
```

## Go packages

```bash
go test ./...
```

Build/install the native library first when a package requires native linking.

## CUDA build

```bash
cmake -S . -B build-cuda -DCMAKE_BUILD_TYPE=Release -DPYGO_ENABLE_CUDA=ON -DPYGO_BUILD_SHARED=ON -DPYGO_BUILD_TESTS=ON
cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```
