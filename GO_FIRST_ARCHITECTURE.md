# PyGo Go-First Architecture

PyGo has one public programming language: **Go**.  Python is embedded inside
the Go process for autograd, the model zoo, research tooling, and Python
interoperability.  C++/CUDA is the native compute engine.

## Production execution graph

```text
Go application
      |
      v
PyGo Go API
      |
     CGo
      |
      v
Stable C ABI
      |
      v
Embedded CPython
      |
      v
pybind11: _pygo_native
      |
      v
C++ native engine
      |
      +-------------------+
      |                   |
      v                   v
 CUDA kernels        CPU kernels
                     AVX-512 / scalar
```

There is **no Python sidecar process** in the normal embedded execution path.
CPython is initialized by the Go process and the pybind11 extension is
registered into that interpreter before initialization.

## Why Python remains a core folder

The `python/` tree is not a compatibility stub.  It provides the Python
implementation ecosystem used inside the embedded interpreter:

```text
python/
├── autograd/       reverse-mode/autograd integration
├── models/         model zoo
├── nn/             neural-network components
├── optim/          optimizers
├── data/           datasets and loaders
├── training/       training orchestration
├── inference/      inference helpers
├── serialization/  checkpoints and model formats
├── distributed/    distributed training
├── runtime/        embedded runtime helpers
└── bridge/         Python-side native bridge utilities
```

Go remains the public API; Python supplies capabilities rather than becoming a
second required application language.

## Zero-copy tensor model

Tensor bytes are stored in native memory and referenced through opaque handles.
Go and Python hold references to the same allocation:

```text
                         Native storage
                              |
             +----------------+----------------+
             |                |                |
        Go Tensor        C++ Tensor       Python capsule
             |                |                |
             +----------------+----------------+
                       same storage
```

The CGo bridge creates a reference-counted `PyGoTensor` descriptor.  The Go
wrapper owns one reference and `ToPython()` gives the CPython capsule another.
The capsule destructor releases only its reference, so Python cannot observe a
freed descriptor merely because the Go wrapper was garbage-collected.

## GIL and interpreter rules

All CPython C-API calls are protected by a GIL guard.  CPython is initialized
using `PyConfig` rather than legacy global initialization calls.  Native
pybind11 modules are registered before `Py_InitializeFromConfig`.

## Native compute ownership

The native C++ layer remains authoritative for tensor representation and
numerical execution.  Python-side objects should carry handles/metadata and
invoke the native engine instead of maintaining a second independent tensor
storage implementation.

CUDA execution additionally requires stream/event synchronization.  A shared
pointer is not a substitute for device synchronization.

## Build modes

```text
PYGO_ENABLE_PYTHON=AUTO|ON|OFF
PYGO_ENABLE_PYBIND=AUTO|ON|OFF
PYGO_ENABLE_CUDA=AUTO|ON|OFF
```

A complete embedded build uses Python development/embed files and pybind11.
Setting the Python and pybind options to `ON` turns missing dependencies into
configuration errors rather than silently producing a different architecture.
