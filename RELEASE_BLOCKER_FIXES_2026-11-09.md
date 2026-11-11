# PyGo Fast Release — Critical Bridge Fixes

Date: 2026-11-09

This patch addresses the three concrete bridge/runtime defects identified in the embedded Go + CPython + pybind11 architecture.

## 1. Python exception lifetime

The old error path returned a borrowed UTF-8 pointer after decrementing the Python string object. The pointer could become invalid, producing `unknown error` or worse.

The bridge now copies the exception message into a caller-provided buffer before releasing Python objects. `RunPython()` also uses `PyRun_StringFlags()` so the Python exception remains available to the error extractor.

Verified with:

`raise RuntimeError('bridge-error-579')`

Expected and observed Go error:

`Python error: bridge-error-579`

## 2. CPython lifecycle state

The old `sync.Once` state could report successful reinitialization after `FinalizePython()` even though CPython was no longer initialized.

The bridge now protects lifecycle state with a mutex and explicitly rejects reinitialization after finalization. This is deliberate because embedded extension modules can retain interpreter-global state across CPython finalization.

Verified sequence:

`InitPython -> IsInitialized -> FinalizePython -> IsInitialized=false -> InitPython returns error`

## 3. Native C++ Tensor ownership for bridge-created CPU float32 tensors

Bridge-created CPU float32 tensors now use the production C++ Tensor C ABI when those symbols are linked into the Go process. The C++ Tensor handle owns the native storage; the Go/Python capsule retains the bridge metadata and reference-counted native handle.

This removes the previous split allocation model for the primary float32 CPU path:

`Go -> CGo -> C++ Tensor storage -> Python capsule`

No intermediate copy is introduced.

The CGo declarations are weak so the Go package remains testable by itself; when `libpygo` is linked, the native C++ implementation is used.

The pybind11 module now also carries the native handle in its capsule ABI and exposes `native_engine` plus `tensor_native_sum` for integration validation.

## Additional leak fix

`pygo_call_function()` now decrements the temporary zero-argument tuple it creates.

## Verification

- `go test ./bindings/cgo`: PASS
- embedded CPython tests: PASS
- Python exception preservation test: PASS
- CPython finalize/reinitialize lifecycle executable: PASS
- CMake `pygo_c_api_obj`: PASS
- full `pygo_native` build: started successfully but was terminated by the execution time limit after substantial compilation; no compiler error was observed before termination
- CUDA: not re-run locally; the project's previous Colab T4 result remains 85/85 `.cu` compilation passes
- pybind11: not compiled in this environment because pybind11 headers are not installed

No `cpp/native_kernels` directory was added.
