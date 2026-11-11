# PyGo Embedded CPython Implementation Report — 2026-11-09

## Architecture implemented

PyGo now has an explicit in-process architecture for the Go-first runtime:

```text
Go API
  -> CGo
  -> stable/native C boundary
  -> embedded CPython
  -> statically linked pybind11 extension (_pygo_native)
  -> C++ native engine
  -> CUDA / CPU kernels
```

Python is not launched as a subprocess in this design.

## Native changes

- Added `cpp/bridge/embedded_python.h`.
- Added `cpp/bridge/embedded_python.cpp`.
- Added `bindings/pybind/module.cpp` with a real `_pygo_native` pybind11 module.
- Added CMake controls:
  - `PYGO_ENABLE_PYTHON=AUTO|ON|OFF`
  - `PYGO_ENABLE_PYBIND=AUTO|ON|OFF`
  - `PYGO_PYBIND_INCLUDE_DIR=<path>` fallback for header-only pybind11 installs.
- CMake automatically asks `python -m pybind11 --cmakedir` for the pybind11 CMake package when available.

## Zero-copy bridge changes

The CGo `BridgeTensor` path was hardened so the shared descriptor and storage
are reference-counted on the native side.

- Tensor bytes are allocated in native memory.
- Go keeps a native descriptor reference.
- `ToPython()` gives the Python capsule an additional reference.
- Capsule destruction releases only the Python reference.
- The native allocation is freed only after the final reference disappears.
- Shape metadata supports zero-rank tensors without indexing an empty Go slice.
- Negative tensor dimensions are rejected.

This removes the previous lifetime hazard where a Python capsule could outlive
the Go wrapper while the descriptor/storage was still assumed to be Go-owned.

## Python-side API

Added `python/bridge/embedded.py` and exposed runtime capability reporting from
`python/runtime/__init__.py`.

The Python bridge can report whether `_pygo_native` is available and can create
a zero-copy `TensorView` from a `PyGoTensor` capsule.

## Verification performed in this environment

- Python bytecode compilation: PASS for `python/` and `bindings/`.
- `go test ./bindings/cgo`: PASS.
- `go test ./tensor`: PASS.
- CPython initialization/run probe through CGo: PASS.
- `embedded_python.cpp` standalone C++ compilation against Python 3.13: PASS.
- `bindings/pybind/module.cpp` standalone compilation against the available
  pybind11 headers: PASS.
- CMake configuration with embedded CPython enabled: PASS.
- CUDA translation-unit count: 85.
- `cpp/native_kernels`: absent.

## Important verification boundary

The current host does not have an NVIDIA CUDA compiler, so this change was not
claimed to be a complete CUDA runtime validation.  The user-provided Colab
T4 validation separately reported 85/85 CUDA source files passing `nvcc`.

The local host also does not have a standalone pybind11 CMake package installed;
its available PyTorch distribution supplies pybind11 headers, which were used
to compile-check the new module.  A final release build should install a
supported pybind11 package and run the complete CMake build with
`PYGO_ENABLE_PYTHON=ON` and `PYGO_ENABLE_PYBIND=ON`.
