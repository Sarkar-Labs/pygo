# Contributing to PyGo

Thank you for your interest in contributing to **PyGo**—a deep learning framework combining a Go API, embedded CPython autograd, and high-performance C++/CUDA kernels.

## Architecture Overview

Before submitting code, please familiarize yourself with how data flows across the three layers:

```text
Go API → CGo Bridge → CPython (Embedded) → pybind11 → CUDA / AVX-512 Kernels
```

- **Zero-Copy Shared Memory:** Tensors live in shared memory across Go, Python, and C++. Avoid unnecessary memory allocations or copies across language boundaries.
- **Boundary Overhead:** Pay close attention to CGo call overhead when designing new Go-to-Python tensor operations.

## Areas Where We Need Help

We are actively looking for contributors across the stack:
1. **Go & CGo Engineers:** Optimizing the CGo bridge, reducing tensor boundary overhead, and building idiomatic Go APIs with goroutine-based dynamic batching.
2. **Python Engineers:** Expanding the autograd engine, model zoo (LLaMA-3, GPT-2, BERT, ViT, Whisper), and distributed trainers (FSDP, DPO, SFT).
3. **C++ / CUDA Developers:** Writing and optimizing low-level kernels, Flash Attention v2, and GPTQ/AWQ quantization routines.

## Development Setup

### Prerequisites
- **Go:** 1.22+ (with `CGO_ENABLED=1`)
- **Python:** 3.10+ (with development headers `python3-dev` installed)
- **C++ Compiler:** GCC 11+ or Clang 14+ supporting C++17/C++20 and AVX-512
- **CUDA Toolkit:** 12.0+ (for GPU kernel development)
- **Build Tools:** CMake 3.20+ and `pybind11`

### Building from Source
1. Fork the repository and clone it locally:
   ```bash
   git clone https://github.com/Sarkar-Labs/pygo.git
   cd pygo
   ```
2. Build the C++/CUDA backend and `pybind11` extensions:
   ```bash
   mkdir -p build && cd build
   cmake .. -DUSE_CUDA=ON
   make -j\$(nproc)
   cd ..
   ```
3. Verify the Go and CGo bindings:
   ```bash
   CGO_ENABLED=1 go test ./...
   ```

## Pull Request Process

1. **Create a Branch:** Use a descriptive branch name (e.g., `feat/cgo-tensor-pool` or `fix/cuda-flash-attn`).
2. **Code Formatting:**
   - Run `gofmt -s -w .` for Go files.
   - Format Python code with `black` or `ruff`.
   - Format C++/CUDA files using `clang-format`.
3. **Tests & Benchmarks:** Include unit tests for any new operators or layers. If modifying the CGo bridge or CUDA kernels, include before-and-after latency/throughput benchmarks.
4. **Submit PR:** Open a pull request against `main` with a clear summary of the problem solved, architectural impact, and benchmark results.

## Communication & Support

- Open a GitHub Issue to discuss major architectural changes or report bugs before writing large amounts of code.
- Check existing issues labeled `good first issue` or `help wanted` to get started.
