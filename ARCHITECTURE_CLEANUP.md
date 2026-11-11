# PyGo Architecture Cleanup

## Why `cpp/native_kernels` is not part of PyGo

PyGo's native implementation is organized by responsibility: tensor operations live in `cpp/tensor`, neural-network primitives in `cpp/nn`, automatic differentiation in `cpp/autograd`, optimization in `cpp/optim`, inference in `cpp/inference`, distributed functionality in `cpp/distributed`, CUDA implementations in `cpp/cuda`, and shared runtime/device facilities in their respective directories.

The repository does not use a synthetic `cpp/native_kernels` directory to increase source-line counts. Operator implementations belong in the subsystem that owns their semantics, with public headers, dispatch, tests, and backend implementations wired through the normal build.

## Verification after cleanup

The CPU production build was configured with CUDA disabled and the native test suite completed successfully:

- CMake configure: passed
- Native build: passed
- CTest: 2/2 passed
- Go `go test ./...`: passed

CUDA remains hardware/toolchain dependent and must be compiled with a real NVIDIA CUDA Toolkit using the repository's CUDA validation workflow.
