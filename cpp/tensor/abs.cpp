#include "pygo/abs.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

Tensor abs(const Tensor& input) {
    auto output = zeros(input.shape(), input.dtype(), input.device());
    
    if (input.device().is_cpu()) {
        int64_t total_elements = input.numel();
        
        if (input.dtype() == DType::Float32) {
            const float* src = input.data<float>();
            float* dst = output.data<float>();

            #pragma omp parallel for if(total_elements > 65536)
            for (int64_t i = 0; i < total_elements; ++i) {
#if defined(__AVX2__)
                // Process 8 floats at a time using AVX2 bitwise clearing of sign bit
                for (; i <= total_elements - 8; i += 8) {
                    __m256 v = _mm256_loadu_ps(src + i);
                    // Clear the sign bit (mask out the highest bit of each 32-bit float)
                    __m256 mask = _mm256_set1_ps(-0.0f); // 0x80000000
                    __m256 res = _mm256_andnot_ps(mask, v);
                    _mm256_storeu_ps(dst + i, res);
                }
#elif defined(__ARM_NEON)
                for (; i <= total_elements - 4; i += 4) {
                    float32x4_t v = vld1q_f32(src + i);
                    float32x4_t res = vabsq_f32(v);
                    vst1q_f32(dst + i, res);
                }
#endif
                if (i < total_elements) {
                    dst[i] = std::abs(src[i]);
                }
            }
        } else {
            throw std::runtime_error("pygo::abs currently optimized primarily for Float32 data type.");
        }
    } else {
        throw std::runtime_error("pygo::abs acceleration not yet dispatched for non-CPU devices.");
    }
    
    return output;
}

void abs_(Tensor& input) {
    if (input.device().is_cpu()) {
        int64_t total_elements = input.numel();
        if (input.dtype() == DType::Float32) {
            float* data = input.data<float>();

            #pragma omp parallel for if(total_elements > 65536)
            for (int64_t i = 0; i < total_elements; ++i) {
                data[i] = std::abs(data[i]);
            }
        } else {
            throw std::runtime_error("pygo::abs_ in-place currently supports Float32.");
        }
    } else {
        throw std::runtime_error("pygo::abs_ in-place not yet implemented for non-CPU devices.");
    }
}

} // namespace pygo