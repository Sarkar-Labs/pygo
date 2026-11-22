# PyGo 0.1.0

PyGo is a deep-learning framework designed with **Go as the top-level application API**. Python provides the autograd and model ecosystem, while C++ and CUDA provide the native tensor runtime and raw compute.

## Architecture

```text
Go API
   │
   ▼
CGo / stable C ABI
   │
   ▼
Embedded CPython
   │
   ▼
pybind11
   │
   ▼
C++ native runtime
   │
   ├── CPU / AVX-512
   └── CUDA
```

The important design choice is that CPython is intended to run **inside the Go process**, not as a Python sidecar. The native runtime owns tensor storage, allowing the language layers to exchange tensor handles and compatible buffers without unnecessary serialization or process boundaries.

## What each layer does

| Layer | Responsibility |
|---|---|
| **Go** | Public API, application integration, training/inference orchestration, serving |
| **CGo / C ABI** | Stable boundary between Go and native runtime |
| **Embedded CPython** | In-process Python runtime for the Python ecosystem |
| **Python** | Autograd integration, model zoo, high-level utilities and callbacks |
| **pybind11** | Native C++ ↔ Python bindings |
| **C++** | Tensor runtime, memory management, operators and dispatch |
| **CUDA / AVX-512** | Hardware-specific compute kernels |

## Tensor memory model

PyGo is designed around shared native tensor storage:

```text
          ┌───────────────────────┐
          │   Native Tensor       │
          │ metadata + storage    │
          └───────────┬───────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
        Go API      Python      C++/CUDA
```

A zero-copy path is only valid when the storage lifetime and ownership contract are explicit. PyGo therefore distinguishes normal copying constructors from external-buffer APIs.

## Repository layout

```text
pygo/
├── go/                 # Go public API and Go ecosystem
├── python/             # Python autograd/model ecosystem
├── cpp/                # C++ runtime, CPU/AVX and CUDA implementation
├── bindings/            # CGo/CFFI/pybind11/SWIG binding sources
├── bridge/              # Cross-language bridge logic
├── docs/                # Architecture, guides and API documentation
├── tests/               # Cross-language and integration tests
├── .github/workflows/   # CI, CUDA validation and release automation
├── CMakeLists.txt
├── go.mod
├── pyproject.toml
└── VERSION
```

## Installation

### Build the native CPU runtime

```bash
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYGO_ENABLE_CUDA=OFF \
  -DPYGO_BUILD_SHARED=ON \
  -DPYGO_BUILD_TESTS=ON

cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

### Python environment

```bash
python -m pip install pygo-python -e .
python -m pytest tests/python -q
```

### Go

```bash
go test ./...
```

The Go packages that use the native runtime require the PyGo native library to be built and discoverable by the linker/runtime loader.

## CUDA

CUDA is optional in 0.1.0. A CUDA-capable development machine can configure the project with:

```bash
cmake -S . -B build-cuda \
  -DCMAKE_BUILD_TYPE=Release \
  -DPYGO_ENABLE_CUDA=ON \
  -DPYGO_BUILD_SHARED=ON \
  -DPYGO_BUILD_TESTS=ON

cmake --build build-cuda --parallel
ctest --test-dir build-cuda --output-on-failure
```

The repository also contains a dedicated CUDA GitHub Actions workflow for a self-hosted NVIDIA runner. GitHub-hosted CPU runners are not presented as CUDA hardware validation.

## Release status

**PyGo 0.1.0 is the initial release line.** The release documentation intentionally separates implemented source from hardware/runtime validation. CUDA, AVX-512, embedded CPython and other optional components should be considered supported on a target only after the corresponding build and runtime tests pass on that target.

See:

- [`docs/architecture/ARCHITECTURE.md`](docs/architecture/ARCHITECTURE.md)
- [`docs/architecture/EMBEDDED_CPYTHON.md`](docs/architecture/EMBEDDED_CPYTHON.md)
- [`docs/guides/installation.md`](docs/guides/installation.md)
- [`docs/guides/quickstart.md`](docs/guides/quickstart.md)
- [`docs/RELEASE_0.1.0.md`](docs/RELEASE_0.1.0.md)
- [`CHANGELOG.md`](CHANGELOG.md)

## License

PyGo 0.1.0 is released under the MIT License. See [`LICENSE`](LICENSE).
