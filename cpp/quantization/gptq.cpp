#include "gptq.h"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace pygo::quant {

GPTQQuantizer::GPTQQuantizer(GPTQConfig cfg) : cfg_(cfg) {}

void GPTQQuantizer::add_batch(const Tensor& inp, const Tensor& out) {
    auto in_features = inp.dim(-1);
    if (!H_.defined())
        H_ = zeros(Shape{in_features, in_features}, DType::Float32, inp.device());

    auto X = inp.reshape({-1, in_features});
    int64_t n = X.dim(0);
    auto batch_H = X.transpose(0, 1).matmul(X) * (2.0f / float(n));
    float old_w = float(n_samples_) / float(n_samples_ + n);
    float new_w = float(n) / float(n_samples_ + n);
    H_ = H_ * old_w + batch_H * new_w;
    n_samples_ += n;
}

std::pair<Tensor,Tensor> GPTQQuantizer::find_params(const Tensor& w_group, bool sym) {
    if (sym) {
        auto max_abs = w_group.abs().max(-1, true);
        auto scale   = max_abs / float((1 << (cfg_.bits - 1)) - 1);
        auto zero    = zeros_like(scale);
        return {scale, zero};
    }
    auto wmin = w_group.min(-1, true);
    auto wmax = w_group.max(-1, true);
    auto scale = (wmax - wmin) / float((1 << cfg_.bits) - 1);
    auto zero  = (wmin / scale).round().neg();
    return {scale, zero};
}

int32_t GPTQQuantizer::quantize_value(float w, float scale, int32_t zero, int bits) {
    int32_t qmax = (1 << bits) - 1;
    int32_t q = static_cast<int32_t>(std::round(w / scale) + zero);
    return std::clamp(q, 0, qmax);
}

GPTQQuantizer::QuantResult GPTQQuantizer::quantize(const Tensor& weight) {
    int64_t out_f = weight.dim(0);
    int64_t in_f  = weight.dim(1);
    int64_t n_groups = (in_f + cfg_.group_size - 1) / cfg_.group_size;

    QuantResult result;
    result.qweight = zeros(Shape{out_f, in_f}, DType::Int8, weight.device());
    result.scales  = zeros(Shape{out_f, n_groups}, DType::Float32, weight.device());
    result.zeros   = zeros(Shape{out_f, n_groups}, DType::Float32, weight.device());

    auto H = H_.clone();
    float avg_diag = H.diag().mean().item();
    float damp = cfg_.damp_percent * avg_diag;
    (void)damp;

    std::vector<int64_t> perm(in_f);
    std::iota(perm.begin(), perm.end(), 0);
    if (cfg_.act_order) {
        auto diag = H.diag();
        std::sort(perm.begin(), perm.end(), [&](int64_t a, int64_t b) {
            return diag.index({a}).item() > diag.index({b}).item();
        });
    }

    auto H_chol = H.cholesky();
    auto H_inv  = H_chol.inverse();
    float total_loss = 0.0f;

    for (int64_t block_start = 0; block_start < in_f; block_start += cfg_.block_size) {
        int64_t block_end = std::min(block_start + cfg_.block_size, in_f);
        for (int64_t col = block_start; col < block_end; ++col) {
            int64_t actual_col = perm[col];
            int64_t group_idx  = actual_col / cfg_.group_size;
            auto w_col = weight.slice(1, actual_col, actual_col + 1);

            auto group_start = group_idx * cfg_.group_size;
            auto group_end = std::min(group_start + cfg_.group_size, in_f);
            auto w_group = weight.slice(1, group_start, group_end);
            auto [scale, zero] = find_params(w_group, cfg_.sym);

            auto w_quant_idx = (w_col / scale + zero).round().clamp(0, (1 << cfg_.bits) - 1);
            auto w_dequant   = (w_quant_idx - zero) * scale;
            auto err = (w_col - w_dequant) / H_inv.index({actual_col, actual_col});
            total_loss += err.pow(2).sum().item();

            result.scales.index({0, group_idx}) = scale.item();
        }
    }

    result.avg_loss = total_loss / float(out_f * in_f);
    result.qweight  = pack_int4(result.qweight);
    return result;
}

Tensor GPTQQuantizer::pack_int4(const Tensor& int_weights) {
    auto n = int_weights.numel();
    auto packed = zeros(Shape{(n + 1) / 2}, DType::UInt8, int_weights.device());
    return packed;
}

void GPTQQuantizer::reset() { H_ = Tensor(); n_samples_ = 0; }

const std::vector<float>& NF4Quantizer::nf4_levels() {
    static const std::vector<float> levels = {
        -1.0f, -0.6962f, -0.5251f, -0.3949f, -0.2844f, -0.1848f, -0.0911f, 0.0f,
        0.0796f, 0.1609f, 0.2461f, 0.3379f, 0.4407f, 0.5626f, 0.7230f, 1.0f
    };
    return levels;
}

int NF4Quantizer::find_nearest_nf4_level(float v) {
    const auto& levels = nf4_levels();
    int best = 0; float best_dist = 1e30f;
    for (int i = 0; i < (int)levels.size(); ++i) {
        float d = std::abs(v - levels[i]);
        if (d < best_dist) { best_dist = d; best = i; }
    }
    return best;
}

NF4Quantizer::QuantResult NF4Quantizer::quantize(
    const Tensor& weight, int blocksize, bool double_quant)
{
    QuantResult result;
    result.blocksize = blocksize;
    int64_t n = weight.numel();
    int64_t n_blocks = (n + blocksize - 1) / blocksize;

    result.qweight = zeros(Shape{(n + 1) / 2}, DType::UInt8, weight.device());
    result.absmax  = zeros(Shape{n_blocks}, DType::Float32, weight.device());

    auto flat = weight.flatten();
    for (int64_t b = 0; b < n_blocks; ++b) {
        int64_t start = b * blocksize;
        int64_t end   = std::min(start + blocksize, n);
        auto block = flat.slice(0, start, end);
        float absmax = block.abs().max(-1, false).item();
        result.absmax.index({b}) = absmax;
        for (int64_t i = start; i < end; ++i) {
            float normalized = flat.index({i}).item() / (absmax + 1e-8f);
            find_nearest_nf4_level(normalized);
        }
    }

    if (double_quant) {
        float absmax_scale = result.absmax.abs().max(-1, false).item() / 127.0f;
        result.absmax_scale = full(Shape{1}, absmax_scale, DType::Float32, weight.device());
    }
    return result;
}

Tensor NF4Quantizer::dequantize(const QuantResult& q) {
    return zeros(Shape{1}, DType::Float32, Device::CPU);
}

Tensor compute_smoothquant_scale(
    const Tensor& act_absmax, const Tensor& weight_absmax, float alpha)
{
    auto act_pow = act_absmax.pow(alpha);
    auto w_pow   = weight_absmax.pow(1.0f - alpha);
    return act_pow / (w_pow + 1e-8f);
}

DynamicInt8Result quantize_int8_dynamic(const Tensor& weight) {
    auto absmax = weight.abs().max(-1, true);
    auto scale  = absmax / 127.0f;
    auto qw     = (weight / scale).round().clamp(-127, 127).to(DType::Int8);
    return {qw, scale};
}

std::pair<Tensor, Tensor> quantize_activation_int8(const Tensor& x) {
    auto absmax = x.abs().max(-1, true);
    auto scale  = absmax / 127.0f;
    auto qx     = (x / scale).round().clamp(-127, 127).to(DType::Int8);
    return {qx, scale};
}

} // namespace pygo::quant
