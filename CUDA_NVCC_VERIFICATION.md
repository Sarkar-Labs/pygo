# PyGo CUDA / nvcc verification

## Scope

PyGo contains 80 CUDA translation units under `cpp/cuda`. This release pass audited all 80 files for build-critical problems and corrected the CUDA public header/runtime integration, declaration/definition mismatches, accidental Markdown citation artifacts, invalid kernel return statements, and several placeholder-quality kernels.

## What was verified in this environment

The current environment does **not** contain NVIDIA `nvcc`, a CUDA toolkit, or a CUDA GPU. Therefore a real NVIDIA compilation cannot honestly be claimed here.

The following checks do run locally:

- 80/80 `.cu` files discovered.
- 0 empty `.cu` files.
- 0 `.cu` files with 10 lines or fewer.
- 79/80 files contain CUDA kernel launch syntax; the remaining translation unit uses CUDA APIs through its implementation path.
- 0 citation/code-fence artifacts remain in CUDA sources.
- CUDA header/function declaration vs definition parameter counts are consistent.
- CPU CMake configuration with `PYGO_ENABLE_CUDA=AUTO` succeeds.
- CPU build succeeds after the CUDA integration changes.
- CTest passes 2/2 CPU tests.

## Real nvcc validation

Run this on an NVIDIA CUDA development machine:

```bash
python tools/cuda/validate_cuda_sources.py --nvcc /usr/local/cuda/bin/nvcc
```

The validator compiles **every `.cu` file individually with nvcc** using C++17 and the PyGo include paths. It reports the exact file if compilation fails.

A successful CUDA validation ends with:

```text
CUDA nvcc compilation: PASS (80/80 files)
```

## CMake CUDA build

Configure:

```bash
cmake -S . -B build-cuda -DPYGO_ENABLE_CUDA=ON
```

Build:

```bash
cmake --build build-cuda --parallel
```

Or validate every CUDA translation unit independently:

```bash
cmake --build build-cuda --target pygo_cuda_validate
```

`PYGO_ENABLE_CUDA=AUTO` remains safe for machines without CUDA: it selects the CPU backend instead of pretending that CUDA is available.

## Important release rule

This repository now contains a real CUDA build path, but CUDA hardware execution is only certified after the 80-file `nvcc` validation and GPU runtime tests have been run on a CUDA-capable machine or CI runner. This document intentionally does not claim that a local CPU-only environment compiled CUDA code.
