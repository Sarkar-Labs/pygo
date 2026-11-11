#include "pygo/block_reduce.h"
#include <stdexcept>
#include <cmath>
#include <numeric>
#include <algorithm>

namespace pygo {

Tensor reduce_sum(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::reduce_sum is currently supported only on CPU.");
    }

    Tensor input_c = input.contiguous();
    int64_t numel = input_c.numel();

    if (dims.empty()) {
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(input.ndim(), 1) : std::vector<int64_t>{1};
        Tensor out = empty(Shape(out_shape), input.dtype(), Device::CPU);

        if (input.dtype() == DataType::Float32) {
            float val = 0.0f;
            const float* data = input_c.data<float>();
            #pragma omp parallel for reduction(+:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                val += data[i];
            }
            *out.mutable_data<float>() = val;
        } else if (input.dtype() == DataType::Float64) {
            double val = 0.0;
            const double* data = input_c.data<double>();
            #pragma omp parallel for reduction(+:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                val += data[i];
            }
            *out.mutable_data<double>() = val;
        } else if (input.dtype() == DataType::Int32) {
            int32_t val = 0;
            const int32_t* data = input_c.data<int32_t>();
            #pragma omp parallel for reduction(+:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                val += data[i];
            }
            *out.mutable_data<int32_t>() = val;
        } else if (input.dtype() == DataType::Int64) {
            int64_t val = 0;
            const int64_t* data = input_c.data<int64_t>();
            #pragma omp parallel for reduction(+:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                val += data[i];
            }
            *out.mutable_data<int64_t>() = val;
        } else {
            throw std::runtime_error("Unsupported DataType for pygo::reduce_sum.");
        }
        return out;
    }

    throw std::runtime_error("pygo::reduce_sum with specific dims is not yet implemented.");
}

Tensor reduce_mean(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim) {
    Tensor sum_tensor = reduce_sum(input, dims, keepdim);
    int64_t reduced_elements = 1;
    if (dims.empty()) {
        reduced_elements = input.numel();
    } else {
        for (auto d : dims) {
            reduced_elements *= input.shape().dims[d];
        }
    }

    if (reduced_elements == 0) return sum_tensor;

    Tensor out = sum_tensor.clone();
    int64_t numel = out.numel();
    if (out.dtype() == DataType::Float32) {
        float* ptr = out.mutable_data<float>();
        float scale = 1.0f / static_cast<float>(reduced_elements);
        #pragma omp parallel for if(numel > 65536)
        for (int64_t i = 0; i < numel; ++i) ptr[i] *= scale;
    } else if (out.dtype() == DataType::Float64) {
        double* ptr = out.mutable_data<double>();
        double scale = 1.0 / static_cast<double>(reduced_elements);
        #pragma omp parallel for if(numel > 65536)
        for (int64_t i = 0; i < numel; ++i) ptr[i] *= scale;
    } else {
        throw std::runtime_error("reduce_mean on integer types requires float accumulation/output.");
    }
    return out;
}

Tensor reduce_max(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::reduce_max is currently supported only on CPU.");
    }
    Tensor input_c = input.contiguous();
    int64_t numel = input_c.numel();
    if (dims.empty()) {
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(input.ndim(), 1) : std::vector<int64_t>{1};
        Tensor out = empty(Shape(out_shape), input.dtype(), Device::CPU);
        if (numel == 0) throw std::runtime_error("reduce_max on empty tensor");

        if (input.dtype() == DataType::Float32) {
            float val = input_c.data<float>()[0];
            const float* data = input_c.data<float>();
            #pragma omp parallel for reduction(max:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] > val) val = data[i];
            }
            *out.mutable_data<float>() = val;
        } else if (input.dtype() == DataType::Float64) {
            double val = input_c.data<double>()[0];
            const double* data = input_c.data<double>();
            #pragma omp parallel for reduction(max:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] > val) val = data[i];
            }
            *out.mutable_data<double>() = val;
        } else if (input.dtype() == DataType::Int32) {
            int32_t val = input_c.data<int32_t>()[0];
            const int32_t* data = input_c.data<int32_t>();
            #pragma omp parallel for reduction(max:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] > val) val = data[i];
            }
            *out.mutable_data<int32_t>() = val;
        } else if (input.dtype() == DataType::Int64) {
            int64_t val = input_c.data<int64_t>()[0];
            const int64_t* data = input_c.data<int64_t>();
            #pragma omp parallel for reduction(max:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] > val) val = data[i];
            }
            *out.mutable_data<int64_t>() = val;
        } else {
            throw std::runtime_error("Unsupported DataType for pygo::reduce_max.");
        }
        return out;
    }
    throw std::runtime_error("pygo::reduce_max with specific dims is not yet implemented.");
}

Tensor reduce_min(const Tensor& input, const std::vector<int64_t>& dims, bool keepdim) {
    if (!input.device().is_cpu()) {
        throw std::runtime_error("pygo::reduce_min is currently supported only on CPU.");
    }
    Tensor input_c = input.contiguous();
    int64_t numel = input_c.numel();
    if (dims.empty()) {
        std::vector<int64_t> out_shape = keepdim ? std::vector<int64_t>(input.ndim(), 1) : std::vector<int64_t>{1};
        Tensor out = empty(Shape(out_shape), input.dtype(), Device::CPU);
        if (numel == 0) throw std::runtime_error("reduce_min on empty tensor");

        if (input.dtype() == DataType::Float32) {
            float val = input_c.data<float>()[0];
            const float* data = input_c.data<float>();
            #pragma omp parallel for reduction(min:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] < val) val = data[i];
            }
            *out.mutable_data<float>() = val;
        } else if (input.dtype() == DataType::Float64) {
            double val = input_c.data<double>()[0];
            const double* data = input_c.data<double>();
            #pragma omp parallel for reduction(min:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] < val) val = data[i];
            }
            *out.mutable_data<double>() = val;
        } else if (input.dtype() == DataType::Int32) {
            int32_t val = input_c.data<int32_t>()[0];
            const int32_t* data = input_c.data<int32_t>();
            #pragma omp parallel for reduction(min:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] < val) val = data[i];
            }
            *out.mutable_data<int32_t>() = val;
        } else if (input.dtype() == DataType::Int64) {
            int64_t val = input_c.data<int64_t>()[0];
            const int64_t* data = input_c.data<int64_t>();
            #pragma omp parallel for reduction(min:val) if(numel > 65536)
            for (int64_t i = 0; i < numel; ++i) {
                if (data[i] < val) val = data[i];
            }
            *out.mutable_data<int64_t>() = val;
        } else {
            throw std::runtime_error("Unsupported DataType for pygo::reduce_min.");
        }
        return out;
    }
    throw std::runtime_error("pygo::reduce_min with specific dims is not yet implemented.");
}

} // namespace pygo