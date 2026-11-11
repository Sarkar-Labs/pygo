# PyGo 0.1.0 Deep Audit Report

## Scope

This audit checked the release tree for build failures, include-path failures, package-layout problems, Go compilation, Python tests, C++ native build, CTest, and obvious stale references to the removed synthetic `cpp/native_kernels` directory.

## Fixes applied

### C++ include path

`cpp/core/ir.cpp` included `pygo/core.h`, but that public header did not exist. A real IR public header was added at `cpp/include/pygo/core.h`, defining `IRType`, `IRValue`, `IRNode`, and `IRGraph` with the ownership and graph-output semantics required by `ir.cpp`.

### Include audit

All project `pygo/...` includes were checked against the installed public include tree and native source tree.

- Missing `pygo/...` includes before fix: 1
- Missing after fix: 0
- Missing local relative/header includes after fix: 0
- Absolute native include paths: 0

### Go/Python embedding warning

The CGo CPython bridge previously used deprecated `Py_SetProgramName`. It now uses the modern `PyConfig` initialization API and remains compatible with current CPython headers.

### Synthetic kernel cleanup

The artificial `cpp/native_kernels` directory is absent. No build, CI, or packaging target depends on it.

## Verification

- CMake configure: PASS
- Full `pygo_native` C++ build: PASS
- Native smoke test: PASS
- C API smoke test: PASS
- CTest: 2/2 PASS
- `go test ./...`: PASS
- Python compileall: PASS
- Python package-layout tests: 62/62 PASS
- `pygo/...` include audit: PASS
- Local native include audit: PASS
- CUDA `nvcc`: hardware/toolkit dependent and not available in this environment

## Python test invocation

The supported package layout was tested by exposing the repository's `python/` directory as the installed `pygo` package from its parent directory. Directly putting `python/` itself on `PYTHONPATH` is intentionally not a supported layout because it contains `pygo` submodules such as `logging` that can shadow Python's standard-library `logging` module.

## Known semantic work remaining

Some APIs are intentionally abstract or backend-dependent, such as base `Dataset` methods and accelerator-specific transfer paths. They should not be replaced with fake implementations merely to eliminate `NotImplementedError`. CUDA execution still requires a real NVIDIA toolkit/GPU for final runtime certification.
