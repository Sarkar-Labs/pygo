# PyGo CUDA NVCC Fix 2

The Colab NVCC report for the previous archive compiled 83/85 CUDA translation units and failed only:

1. `cpp/cuda/sort_fp32.cu` — `CUDART_INF_F` was undefined. The padding value now uses the device-side IEEE-754 positive-infinity bit pattern via `__int_as_float(0x7f800000)`.
2. `cpp/kernels/elementwise.cu` — `launch_elementwise()` used the C++ conditional operator with `if` statements (`has_b ? if (...) : if (...)`), which is invalid C++. The dispatch now uses ordinary `if/else` blocks around the launch macro.

The previous report showed 83 PASS and 2 FAIL out of 85 CUDA files. The fixes target exactly those two compiler failures.

## Colab verification

Run the full 85-file NVCC compile again with the same `sm_75` command. The expected result after these source fixes is that these two compile errors are gone; the actual PASS count must be established by NVCC on the T4 and should not be assumed without running it.
