# PyGo Embedded CPython Architecture

PyGo is a Go-first deep-learning framework.  Go is the public programming
language and API.  The Python ecosystem is embedded into the same process,
not launched as a sidecar service, and C++/CUDA remains the native compute
engine.

## Runtime graph

```text
Go application
      |
     CGo
      |
Stable native C ABI
      |
Embedded CPython interpreter
      |
pybind11 extension: _pygo_native
      |
C++ native engine
      |
+-----+----------------+
|                      |
CUDA kernels       CPU kernels
(SM75+)            AVX/portable CPU
```

The important property is that the process contains all three runtime layers.
There is no Python subprocess or RPC hop in the normal execution path.

## Tensor ownership

Tensor storage is represented by native allocations and opaque handles.  Go
keeps a handle, Python receives a capsule, and C++ owns the compute/storage
implementation.  The capsule increments the native tensor reference count,
so Python can outlive the Go wrapper without accessing freed storage.

```text
                    Native tensor storage
                           |
             +-------------+-------------+
             |             |             |
         Go handle     C++ Tensor    Python capsule
             |             |             |
             +-------------+-------------+
                     same bytes
```

The Python bridge exposes tensor metadata and the native data address without
copying tensor bytes.  CUDA tensors additionally require stream/event
synchronization; zero-copy does not imply synchronization-free execution.

## Interpreter lifecycle

The Go process calls `InitPython`.  Before `Py_InitializeFromConfig`, the
native library registers `_pygo_native` with `PyImport_AppendInittab`.  CPython
then loads the statically linked pybind11 module from the process itself.

This ordering is intentional: CPython's built-in extension table must be
configured before interpreter initialization.

## pybind11 module

`bindings/pybind/module.cpp` provides the `_pygo_native` extension.  It exposes:

- runtime architecture/capability information;
- a zero-copy `TensorView` metadata object;
- conversion from a `PyGoTensor` capsule to a native view;
- the original native data address for diagnostics.

The module is compiled into `libpygo.so`; it is not loaded from a separate
`.so` at runtime.

## Build configuration

CMake options:

```text
PYGO_ENABLE_PYTHON=AUTO|ON|OFF
PYGO_ENABLE_PYBIND=AUTO|ON|OFF
```

`AUTO` is the default.  `ON` makes the dependency mandatory.

A complete embedded build requires Python development/embed headers and a
CMake-discoverable pybind11 package.  For example, a build machine can install
pybind11 through its normal package manager or Python environment and expose
its CMake directory through `CMAKE_PREFIX_PATH`.

```bash
cmake -S . -B build \
  -DPYGO_ENABLE_PYTHON=ON \
  -DPYGO_ENABLE_PYBIND=ON \
  -DPYGO_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --parallel
```

A CPU-only build can disable CUDA while retaining the embedded Python runtime.

## Go API contract

Go remains the user-facing API.  Applications should not need to know that
CPython is embedded:

```go
package main

import "github.com/pygo"

func main() {
    x := pygo.Randn(pygo.Shape{1024, 1024})
    y := x.Mul(x)
    _ = y
}
```

Python is an implementation/ecosystem layer for autograd integration, the
model zoo, data tooling, serialization, and interoperability.  It is not a
required user-facing language for a Go application.

## Failure and safety rules

1. Never start a Python subprocess for ordinary PyGo execution.
2. Never pass a Go pointer directly into Python after the Go object can be
   collected.  PyGo uses native storage plus a reference-counted capsule.
3. Acquire the GIL around Python API calls.
4. Register built-in pybind11 modules before CPython initialization.
5. Do not treat a zero-copy pointer as synchronization-free on CUDA.
6. Propagate Python exceptions across the CGo boundary instead of silently
   swallowing them in production APIs.
