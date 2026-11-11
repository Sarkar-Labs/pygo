# PyGo 0.1.0 Verification Report

This report records the checks performed in the release workspace. It does not claim accelerator execution on hardware that is not present.

## Verified locally

- CMake configure: PASS (CPU mode)
- Full native C++ build: PASS
- CTest: 2/2 PASS
- Python bytecode compilation: PASS
- Python test suite: 62/62 PASS
- Go test suite: PASS across all Go packages
- Go serialization SafeTensors float32 round-trip: PASS
- C ABI release library refreshed from the current native build
- 83 CUDA translation units present; none are empty or tiny
- 104 Markdown documentation files checked; no short component README remains
- CUDA validation workflow added for a real NVIDIA self-hosted runner

## Accelerator limitation

The current environment has no NVIDIA CUDA toolkit (`nvcc`) and no NVIDIA GPU. Therefore CUDA source compilation and runtime execution are intentionally not reported as passed here.

The repository contains `.github/workflows/cuda.yml`, which performs:

1. `nvidia-smi`
2. `nvcc --version`
3. individual `nvcc` validation of every CUDA translation unit
4. a CUDA-enabled CMake build
5. CTest

Run that workflow on a runner labeled `self-hosted,cuda,nvidia` before publishing a CUDA-verified release.

## Fixes included

- Go tensor float32 data export now uses the stable C ABI data pointer instead of a panic stub.
- Go SafeTensors now writes real float32 tensor bytes and reads them back instead of returning zero tensors.
- Go trainer checkpoint save/load now writes model tensors plus training metadata instead of returning placeholder success.
- CMake/CTest production CPU path remains green after these changes.
