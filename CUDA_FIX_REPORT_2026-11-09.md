# PyGo CUDA Fix Report

Date: 2026-11-09

Source: pygo_0.1.0_deep_fixed.zip

## CUDA compiler failures addressed

### 1. cpp/cuda/matmul_fp16.cu
Removed the illegal child-kernel launch from the `__global__` batched kernel. CUDA dynamic parallelism is not required by PyGo's normal build. Batched dispatch is now performed by a host-side `hgemm_batched` launcher, which invokes the existing WMMA kernel normally.

### 2. cpp/cuda/flash_attention_v2.cu
Reworked the forward kernel to eliminate oversized static shared-memory allocations. The old implementation allocated Q/K/V tiles and a complete score tile in shared memory; D=128 exceeded the SM75/T4 per-block shared-memory limit during ptxas. The new implementation uses the numerically stable online-softmax recurrence and streams K/V from global memory while keeping a per-query O(D) accumulator. It supports the existing FP16, BF16, FP32 and causal/non-causal dispatches for D=64/128.

This is a correctness-first implementation and is expected to trade some performance for broad SM75 compilability and bounded memory use.

### 3. cpp/cuda/sort_fp32.cu
Removed the Thrust dependency that triggered `cuda::std::__4::is_same_v` compatibility errors on the reported Colab CUDA toolchain. Replaced it with a CUDA global-memory bitonic sort using a power-of-two temporary buffer and explicit padding. NaNs are ordered after numeric values.

## Structural checks

- Total `.cu` files in the archive: 85
- `cpp/cuda` `.cu` files: 82
- `cpp/native_kernels`: absent
- Modified CUDA files have balanced braces/parentheses/brackets.
- No child kernel launch remains inside the former batched `__global__` function.
- Local environment does not contain `nvcc`, so final T4 `nvcc` compilation could not be independently executed here.

## Required T4 verification

Run in the user's Colab environment:

    cd /content/pygo
    rm -rf /content/pygo_cuda_build
    mkdir -p /content/pygo_cuda_build

    find cpp/cuda -type f -name '*.cu' -print0 | while IFS= read -r -d '' f; do
      out="/content/pygo_cuda_build/$(echo "$f" | tr '/' '_').o"
      nvcc -std=c++17 -arch=sm_75 -Icpp/include -Icpp -I. -c "$f" -o "$out" || exit 1
    done

Then run the project's CUDA tests and, where an executable/test target is available:

    compute-sanitizer --tool memcheck <test-command>

The archive should not be described as T4-verified until those commands complete successfully.
