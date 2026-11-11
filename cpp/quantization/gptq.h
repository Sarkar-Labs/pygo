#pragma once
#include "pygo/tensor.h"
#include <vector>

namespace pygo::quant {

struct GPTQConfig {
    int     bits          = 4;
    int     group_size    = 128;
    bool    sym            = true;
    bool    act_order      = true;
    bool    true_sequential= true;
    float   damp_percent   = 0.01f;
    int     block_size     = 128;
};

class GPTQQuantizer {
public:
    explicit GPTQQuantizer(GPTQConfig cfg = {});
    void add_batch(const Tensor& inp, const Tensor& out);

    struct QuantResult {
        Tensor qweight;
        Tensor scales;
        Tensor zeros;
        Tensor g_idx;
        float  avg_loss;
    };
    QuantResult quantize(const Tensor& weight);
    void reset();

private:
    GPTQConfig cfg_;
    Tensor     H_;
    int64_t    n_samples_ = 0;
    std::pair<Tensor,Tensor> find_params(const Tensor& w_group, bool sym);
    static int32_t quantize_value(float w, float scale, int32_t zero, int bits);
    static Tensor pack_int4(const Tensor& int_weights);
};

struct AWQConfig {
    int   bits        = 4;
    int   group_size  = 128;
    float zero_point  = true;
    int   n_grid      = 20;
};

class AWQQuantizer {
public:
    explicit AWQQuantizer(AWQConfig cfg = {});
    Tensor compute_activation_scale(const Tensor& activations);
    Tensor search_best_scale(const Tensor& weight, const Tensor& activation_scale,
        const Tensor& calibration_input);
    struct QuantResult { Tensor qweight; Tensor scales; Tensor zeros; };
    QuantResult quantize(const Tensor& weight, const Tensor& act_scale);
private:
    AWQConfig cfg_;
};

class NF4Quantizer {
public:
    static const std::vector<float>& nf4_levels();
    struct QuantResult {
        Tensor qweight; Tensor absmax; Tensor absmax_scale; int blocksize;
    };
    QuantResult quantize(const Tensor& weight, int blocksize = 64, bool double_quant = true);
    Tensor       dequantize(const QuantResult& q);
private:
    int find_nearest_nf4_level(float normalized_val);
};

struct DynamicInt8Result { Tensor qweight; Tensor w_scale; };
DynamicInt8Result quantize_int8_dynamic(const Tensor& weight);
std::pair<Tensor, Tensor> quantize_activation_int8(const Tensor& x);
Tensor compute_smoothquant_scale(const Tensor& activation_absmax,
    const Tensor& weight_absmax, float alpha = 0.5f);

} // namespace pygo::quant
