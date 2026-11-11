#include "focal_loss_fp32.cuh"
#include "pygo/cuda.h"
#include <cuda_runtime.h>
#include <cmath>
#include <stdexcept>

namespace pygo::cuda {
namespace {
__global__ void focal_loss_kernel(const float* logits, const int32_t* targets,
                                  float* result, int64_t rows, int classes,
                                  float gamma, float alpha) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float total = 0.0f;
    for (int64_t r = 0; r < rows; ++r) {
        const int target = targets[r];
        if (target < 0 || target >= classes) continue;
        float max_logit = -CUDART_INF_F;
        for (int c = 0; c < classes; ++c) max_logit = fmaxf(max_logit, logits[r * classes + c]);
        float denom = 0.0f;
        for (int c = 0; c < classes; ++c) denom += expf(logits[r * classes + c] - max_logit);
        const float p = expf(logits[r * classes + target] - max_logit) / fmaxf(denom, 1e-20f);
        const float pt = fminf(fmaxf(p, 1e-8f), 1.0f - 1e-8f);
        total += -alpha * powf(1.0f - pt, gamma) * logf(pt);
    }
    *result = rows > 0 ? total / static_cast<float>(rows) : 0.0f;
}
}

float focal_loss_fp32(const float* logits, const int32_t* targets, int64_t rows,
                      int64_t classes, float gamma, float alpha) {
    if (!logits || !targets || rows < 0 || classes <= 0 || gamma < 0.0f || alpha < 0.0f)
        throw std::invalid_argument("focal_loss_fp32: invalid arguments");
    if (!rows) return 0.0f;
    float* result = nullptr;
    PYGO_CUDA_CHECK(cudaMallocManaged(&result, sizeof(float)));
    focal_loss_kernel<<<1, 1>>>(logits, targets, result, rows, static_cast<int>(classes), gamma, alpha);
    try {
        PYGO_CUDA_CHECK(cudaGetLastError());
        PYGO_CUDA_CHECK(cudaDeviceSynchronize());
        const float value = *result;
        PYGO_CUDA_CHECK(cudaFree(result));
        return value;
    } catch (...) {
        cudaFree(result);
        throw;
    }
}
} // namespace pygo::cuda
