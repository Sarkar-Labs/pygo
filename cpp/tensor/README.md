# PyGo Tensor Architecture

This document explains how `pygo::Tensor` is structured, how tensor memory is represented, how operations are dispatched, how autograd is currently implemented, and how the device model is prepared for CPU, GPU, and XLA/TPU backends.

The most important rule for this document is accuracy: a device enum or API hook is **not** presented as proof that the corresponding hardware backend is already executable.

---

## 1. Architecture overview

The tensor system is built from these core concepts:

```text
Device      -> where the data is stored (CPU, CUDA, MPS, Metal, Vulkan, OpenCL, XLA)
DType       -> the element type (Float32, Int64, Bool, ...)
Storage     -> the raw memory allocation owned by a device
Shape       -> tensor dimensions
Strides     -> mapping from logical indices to storage offsets
TensorImpl  -> Storage + Shape + Strides + dtype + device + gradient metadata
Tensor      -> shared handle around TensorImpl exposed to users
```

The public tensor contract is under `cpp/include/pygo/`, while the current CPU tensor implementation is under `cpp/tensor/`.

---

## 2. Tensor is a lightweight handle

Conceptually, a tensor owns a shared implementation object:

```cpp
struct Tensor {
    std::shared_ptr<TensorImpl> impl_;
};
```

Copying a `Tensor` normally copies the handle rather than the underlying data. A deep copy must explicitly allocate new storage and copy the elements.

Views such as transpose/permute can share storage while changing shape and stride metadata. This is the foundation for zero-copy views inside the tensor runtime.

---

## 3. Device model

`Device` is represented as a `(kind, index)` pair. This keeps the public tensor API independent of a particular accelerator implementation.

```cpp
enum class DeviceKind : uint8_t {
    CPU    = 0,
    CUDA   = 1,   // NVIDIA GPU
    MPS    = 2,   // Apple GPU
    METAL  = 3,   // Apple Metal compute
    VULKAN = 4,   // Vulkan compute
    OPENCL = 5,   // OpenCL accelerator
    XLA    = 6,   // XLA device, including TPU use cases
};
```

For example, `{CUDA, 0}` means the first CUDA device and `{XLA, 0}` identifies the first XLA device.

### Backend status in PyGo 0.1.0

| Device | Status |
|---|---|
| **CPU** | Production-tested native runtime. |
| **CUDA** | CUDA source and CMake integration are present, but CUDA execution is not certified by this CPU-only test environment. |
| **MPS/Metal/Vulkan/OpenCL** | Device/API model exists; no production backend is certified in this release. |
| **XLA/TPU** | Device/API hooks exist, but PyGo 0.1.0 does not bundle a TPU runtime or PJRT backend. |

A device enum must therefore never be interpreted as an assertion that the backend is executable on the current machine.

---

## 4. CUDA build behavior

PyGo uses CMake to make CUDA optional and portable.

The default is:

```bash
cmake -S . -B build
```

which uses:

```text
PYGO_ENABLE_CUDA=AUTO
```

In `AUTO` mode CMake checks whether a CUDA compiler/toolkit is available. If it is not available, PyGo builds the CPU backend instead of failing simply because the machine has no GPU toolchain.

To require CUDA explicitly:

```bash
cmake -S . -B build -DPYGO_ENABLE_CUDA=ON
```

If CUDA is requested explicitly and `nvcc` is missing, configuration fails with a clear diagnostic. This prevents an accidental CPU build from being mistaken for a CUDA build.

To force a CPU-only build:

```bash
cmake -S . -B build -DPYGO_ENABLE_CUDA=OFF
```

The repository's helper build scripts use `AUTO`, so the same scripts can run on CPU-only machines and automatically use CUDA when a compatible toolkit is installed.

You can inspect the local accelerator toolchains with:

```bash
python3 tools/build/detect_accelerators.py
```

---

## 5. TPU/XLA behavior

PyGo 0.1.0 does **not** claim native TPU execution.

The XLA device kind exists so the device model can represent an XLA backend and so future TPU/PJRT integration does not require changing the public tensor representation. Actual TPU execution requires a real XLA/PJRT runtime and backend implementation, neither of which is bundled in this release.

`PYGO_ENABLE_XLA=ON` therefore enables API-level XLA integration hooks and emits a warning that no TPU runtime is bundled. It does not manufacture a fake TPU implementation.

A future TPU backend should provide:

1. device discovery;
2. device allocation and deallocation;
3. host/device transfer;
4. asynchronous execution and synchronization;
5. tensor layout and dtype mapping;
6. operator lowering through XLA/PJRT;
7. error propagation and device lifetime management; and
8. dedicated hardware CI tests.

---

## 6. DType system

The tensor API defines multiple numeric types, including floating point, integer, boolean, complex, and quantized types.

The production CPU path does not automatically imply that every declared dtype is supported by every operation. Operations must validate their dtype requirements and return an explicit error when an implementation is unavailable.

This is preferable to silently interpreting unsupported memory as another dtype.

---

## 7. Shapes and strides

A tensor stores logical dimensions together with stride information.

For a contiguous two-dimensional tensor with shape `[2, 3]`, the row-major strides are `[3, 1]`.

A transpose can change the logical shape and strides without copying storage. That means views can be created without moving the underlying data. Operations that require contiguous storage must explicitly request or create a contiguous representation.

---

## 8. Storage and ownership

Storage represents the allocation behind a tensor. Tensor implementations can share a storage object, which allows multiple views to reference the same memory safely.

This distinction is also important for the public zero-copy C ABI. External buffers must have an explicit ownership/lifetime contract; otherwise a tensor could retain a pointer after the original buffer has been freed.

PyGo's verified C ABI uses retain/release semantics for externally owned buffers rather than assuming that a raw pointer remains valid forever.

---

## 9. Operation execution

Tensor operations generally fall into three categories.

### Elementwise operations

Operations such as add, multiply, ReLU, sigmoid, and similar functions operate element by element and use shape/stride information to locate input values.

### Reductions

Operations such as sum, mean, max, and argmax reduce one or more dimensions. Implementations must preserve the requested `keepdim` semantics and validate dimensions before accessing storage.

### Matrix operations

Matrix multiplication and other linear-algebra operations operate on supported layouts and dtypes. The CPU reference implementation prioritizes correctness and portability; optimized BLAS/SIMD/GPU implementations can be dispatched independently.

---

## 10. Autograd

Autograd associates gradient metadata with tensor computations. A gradient function describes how an operation contributes gradients to its inputs.

The public runtime currently contains both tensor-level gradient metadata and a more general autograd engine under `cpp/autograd/`. These systems are not treated as equivalent merely because both contain a `GradFn`/node concept.

Before enabling advanced graph behavior, the runtime must preserve:

- correct topological traversal;
- gradient accumulation;
- repeated backward semantics;
- graph lifetime;
- higher-order gradient rules; and
- thread/device safety.

Unsupported graph features should fail explicitly rather than return an apparently valid but incorrect gradient.

---

## 11. Per-operation source organization

Many tensor operations have dedicated source files. This makes individual kernels easier to inspect and test, but it also creates a potential duplicate-symbol problem if the same function is implemented in both a central translation unit and a per-operation file.

Production builds therefore need link-time coverage in addition to per-file compilation. A source file compiling successfully by itself does not prove that the entire library links successfully.

---

## 12. Go and Python bindings

### Go

The Go binding uses a C-compatible ABI because cgo cannot directly consume arbitrary C++ classes. PyGo exposes opaque tensor handles and explicit device mapping through `bindings/cgo/`.

This provides a stable boundary:

```text
Go code
   |
   | cgo
   v
C ABI
   |
   v
C++ tensor runtime
```

The Go test suite is part of the release verification process.

### Python

The Python binding can use the native C ABI and/or the C++ binding layer depending on the installed package configuration. Python users should interact with the documented Python API rather than relying on private C++ symbols.

---

## 13. Zero-copy data transfer

Zero-copy has two different meanings in PyGo:

1. **Tensor views:** multiple tensors share the same native storage without copying.
2. **External buffers:** an application gives PyGo a compatible buffer and PyGo wraps it without copying.

For external buffers, pointer identity and lifetime must both be verified. A zero-copy API that has no ownership protocol is unsafe and is not considered production-ready.

The release C ABI includes explicit external-buffer ownership callbacks for this reason.

---

## 14. Hardware verification policy

The development environment used for the PyGo 0.1.0 release does not provide an NVIDIA CUDA toolkit/GPU or a TPU runtime. Therefore the release process performs two kinds of checks:

- **portable checks** that can run on CPU-only machines, such as CMake configuration, CPU builds, ABI tests, and Go/Python tests;
- **accelerator checks** that must run on dedicated CUDA or TPU CI runners before claiming hardware execution support.

For CUDA, the repository now distinguishes:

```text
AUTO -> use CUDA when the toolkit exists, otherwise CPU
ON   -> require CUDA and fail clearly if it is missing
OFF  -> CPU-only build
```

For TPU/XLA, the release only claims API/device-model support until a real PJRT/XLA runtime is integrated and tested on TPU hardware.

---

## 15. Current release summary

| Component | Release status |
|---|---|
| CPU tensor runtime | Production-tested |
| C ABI | Production-tested |
| External-buffer zero-copy path | Tested with ownership callbacks |
| Go binding | Tested with `go test ./...` |
| Python package | Tested in the release environment |
| CUDA source tree | Present and CMake-integrated; hardware execution requires CUDA CI |
| CUDA auto-detection | Implemented |
| TPU/XLA device model | Present |
| TPU runtime | Not bundled |
| TPU hardware execution | Not certified |
| Advanced distributed/accelerator features | Must be validated on their target environments |

This separation between **implemented**, **buildable**, and **hardware-certified** functionality is intentional. It keeps PyGo's release claims reproducible and prevents a device enum, stub, or optional hook from being mistaken for a working accelerator backend.


## 10. Testing the tensor subsystem

Tensor changes should be tested at three levels: API behavior, numerical correctness, and memory/ownership behavior. At minimum, test contiguous tensors, strided views, scalar and empty shapes, broadcasting, dtype validation, invalid dimensions, and device mismatch errors.

From the repository root:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPYGO_ENABLE_CUDA=AUTO
cmake --build build --parallel
ctest --test-dir build --output-on-failure
```

For a CUDA machine, repeat the build with `-DPYGO_ENABLE_CUDA=ON` and run the CUDA-enabled test suite. Do not label a CUDA operator production-ready until it has been executed and compared against a reference implementation.

## 11. Zero-copy views versus zero-copy external buffers

A tensor view and an externally supplied buffer are related but different concepts. A view shares PyGo-owned storage while changing metadata such as shape or strides. An external buffer starts outside the tensor allocator and therefore needs an explicit lifetime callback. Both can avoid data copies, but only the external-buffer API crosses an ownership boundary.

## 12. Practical example

A typical application creates a tensor, applies operations, and passes the result to a higher-level subsystem:

```text
allocate/import storage
        ↓
create TensorImpl
        ↓
validate shape + dtype + device
        ↓
dispatch operation
        ↓
write result storage
        ↓
record autograd metadata when enabled
        ↓
return Tensor handle
```

This separation lets the same tensor contract be used by Python, Go, and native C++ callers while keeping backend-specific memory operations inside the runtime.

## 13. Production checklist

Before changing tensor internals, verify:

- shape and stride calculations are correct;
- no operation reads beyond allocated storage;
- dtype conversions are explicit;
- device transfers are explicit;
- shared storage has a valid lifetime;
- autograd metadata remains consistent;
- errors are propagated instead of ignored; and
- CPU and CUDA implementations agree numerically where both exist.
