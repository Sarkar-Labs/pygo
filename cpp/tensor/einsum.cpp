#include "pygo/einsum.h"
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <stdexcept>
#include <algorithm>
#include <numeric>

#if defined(__AVX2__)
#include <immintrin.h>
#elif defined(__ARM_NEON)
#include <arm_neon.h>
#endif

namespace pygo {

namespace detail {

struct EinsumSpec {
    std::vector<std::string> input_subscripts;
    std::string output_subscript;
};

EinsumSpec parse_equation(const std::string& equation, size_t num_operands) {
    size_t arrow_pos = equation.find("->");
    std::string inputs_part;
    std::string output_part;

    if (arrow_pos == std::string::npos) {
        inputs_part = equation;
        // Implicit mode: alphabetical order of unrepeated indices
        // For simplicity and robustness, explicit equations with '->' are preferred, 
        // but we can handle basic implicit or throw if unsupported.
        throw std::invalid_argument("pygo::einsum requires explicit output specification using '->'.");
    } else {
        inputs_part = equation.substr(0, arrow_pos);
        output_part = equation.substr(arrow_pos + 2);
    }

    std::vector<std::string> inputs;
    std::stringstream ss(inputs_part);
    std::string sub;
    while (std::getline(ss, sub, ',')) {
        // Trim spaces
        sub.erase(std::remove_if(sub.begin(), sub.end(), ::isspace), sub.end());
        inputs.push_back(sub);
    }

    output_part.erase(std::remove_if(output_part.begin(), output_part.end(), ::isspace), output_part.end());

    if (inputs.size() != num_operands) {
        throw std::invalid_argument("Number of input subscripts does not match number of operands in pygo::einsum.");
    }

    return {inputs, output_part};
}

template <typename T>
void einsum_unary_cpu_kernel(const T* src, T* dst, int64_t numel, 
                             const std::vector<int64_t>& reduce_dims) {
    // General reduction/transpose placeholder logic for unary einsum
    #pragma omp parallel for schedule(static) if(numel > 65536)
    for (int64_t i = 0; i < numel; ++i) {
        dst[i] = src[i];
    }
}

template <typename T>
void einsum_matmul_cpu_kernel(const T* a, const T* b, T* dst, 
                              int64_t M, int64_t K, int64_t N) {
    #pragma omp parallel for collapse(2) schedule(static) if(M * N > 1024)
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            T sum = static_cast<T>(0);
            for (int64_t k = 0; k < K; ++k) {
                sum += a[i * K + k] * b[k * N + j];
            }
            dst[i * N + j] = sum;
        }
    }
}

} // namespace detail

Tensor einsum(const std::string& equation, const std::vector<Tensor>& operands) {
    if (operands.empty()) {
        throw std::invalid_argument("pygo::einsum requires at least one operand.");
    }

    for (const auto& op : operands) {
        if (!op.device().is_cpu()) {
            throw std::runtime_error("pygo::einsum is currently supported only on CPU.");
        }
    }

    DType dtype = operands[0].dtype();
    for (size_t i = 1; i < operands.size(); ++i) {
        if (operands[i].dtype() != dtype) {
            throw std::invalid_argument("pygo::einsum expects all operands to have the same dtype.");
        }
    }

    auto spec = detail::parse_equation(equation, operands.size());

    // Validate dimensions match subscript lengths
    for (size_t i = 0; i < operands.size(); ++i) {
        if (operands[i].shape().ndim() != spec.input_subscripts[i].size()) {
            throw std::invalid_argument("Operand dimension does not match subscript length in pygo::einsum.");
        }
    }

    // Handle common binary contraction: "ij,jk->ik" (Matrix Multiplication)
    if (operands.size() == 2 && spec.input_subscripts[0].size() == 2 && 
        spec.input_subscripts[1].size() == 2 && spec.output_subscript.size() == 2) {
        
        char i0 = spec.input_subscripts[0][0];
        char i1 = spec.input_subscripts[0][1];
        char j0 = spec.input_subscripts[1][0];
        char j1 = spec.input_subscripts[1][1];
        char o0 = spec.output_subscript[0];
        char o1 = spec.output_subscript[1];

        if (i1 == j0 && i0 == o0 && j1 == o1) {
            int64_t M = operands[0].shape()[0];
            int64_t K = operands[0].shape()[1];
            int64_t N = operands[1].shape()[1];

            auto output = zeros({M, N}, dtype, operands[0].device());

            if (dtype == DType::Float32) {
                detail::einsum_matmul_cpu_kernel<float>(
                    operands[0].data<float>(), operands[1].data<float>(), 
                    output.data<float>(), M, K, N
                );
            } else if (dtype == DType::Float64) {
                detail::einsum_matmul_cpu_kernel<double>(
                    operands[0].data<double>(), operands[1].data<double>(), 
                    output.data<double>(), M, K, N
                );
            } else {
                throw std::runtime_error("Unsupported DType for pygo::einsum contraction.");
            }

            return output;
        }
    }

    // General fallback for unsupported complex einsum equations
    throw std::runtime_error("The requested einsum contraction pattern is not yet optimized/implemented in pygo::einsum.");
}

} // namespace pygo