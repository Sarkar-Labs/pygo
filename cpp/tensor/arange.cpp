#include "pygo/arange.h"
#include <cmath>
#include <stdexcept>
#include <algorithm>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

void arange_cpu_f32(float* data, int64_t size, double start, double step) {
    const float f_start = static_cast<float>(start);
    const float f_step = static_cast<float>(step);

    #pragma omp parallel for schedule(static) if(size > 65536)
    for (int64_t i = 0; i < size; i += 8) {
#if defined(__AVX2__)
        if (i + 8 <= size) {
            __m256 v_base = _mm256_set1_ps(f_start);
            __m256 v_step = _mm256_set1_ps(f_step);
            __m256 v_offsets = _mm256_set_ps(7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f);
            __m256 v_i = _mm256_set1_ps(static_cast<float>(i));
            
            __m256 v_idx = _mm256_add_ps(v_i, v_offsets);
            __m256 v_val = _mm256_fmadd_ps(v_idx, v_step, v_base);
            
            _mm256_storeu_ps(data + i, v_val);
            continue;
        }
#elif defined(__ARM_NEON)
        if (i + 4 <= size) {
            float32x4_t v_base = vdupq_n_f32(f_start);
            float32x4_t v_step = vdupq_n_f32(f_step);
            float32x4_t v_offsets = {0.0f, 1.0f, 2.0f, 3.0f};
            float32x4_t v_i = vdupq_n_f32(static_cast<float>(i));
            
            float32x4_t v_idx = vaddq_f32(v_i, v_offsets);
            float32x4_t v_val = vmlaq_f32(v_base, v_idx, v_step);
            
            vst1q_f32(data + i, v_val);
            i -= 4; // adjust loop stride step for NEON vector width 4
            continue;
        }
#endif
        // Tail cleanup loop
        for (int64_t j = i; j < std::min(i + 8, size); ++j) {
            data[j] = f_start + static_cast<float>(j) * f_step;
        }
    }
}

template <typename T>
void arange_cpu_generic(T* data, int64_t size, double start, double step) {
    #pragma omp parallel for schedule(static) if(size > 65536)
    for (int64_t i = 0; i < size; ++i) {
        data[i] = static_cast<T>(start + static_cast<double>(i) * step);
    }
}

} // namespace detail

Tensor arange(double start, double end, double step, DType dtype, Device device) {
    if (step == 0.0) {
        throw std::invalid_argument("pygo::arange step must be non-zero.");
    }
    if ((step > 0.0 && start > end) || (step < 0.0 && start < end)) {
        throw std::invalid_argument("pygo::arange upper and lower bounds inconsistent with step direction.");
    }

    int64_t size = static_cast<int64_t>(std::ceil((end - start) / step));
    if (size < 0) size = 0;

    auto output = zeros({size}, dtype, device);
    if (size == 0) return output;

    if (device.is_cpu()) {
        switch (dtype) {
            case DType::Float32:
                detail::arange_cpu_f32(output.data<float>(), size, start, step);
                break;
            case DType::Float64:
                detail::arange_cpu_generic<double>(output.data<double>(), size, start, step);
                break;
            case DType::Int32:
                detail::arange_cpu_generic<int32_t>(output.data<int32_t>(), size, start, step);
                break;
            case DType::Int64:
                detail::arange_cpu_generic<int64_t>(output.data<int64_t>(), size, start, step);
                break;
            default:
                throw std::runtime_error("pygo::arange unsupported data type.");
        }
    } else {
        throw std::runtime_error("pygo::arange hardware acceleration not implemented for non-CPU devices.");
    }

    return output;
}

Tensor arange(double end, DType dtype, Device device) {
    return arange(0.0, end, 1.0, dtype, device);
}

} // namespace pygo