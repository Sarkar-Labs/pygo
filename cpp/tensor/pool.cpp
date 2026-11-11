#include "pygo/pool.h"
#include <stdexcept>
#include <limits>
#include <algorithm>
namespace pygo {

Tensor pool2d(const Tensor& input, std::pair<int64_t,int64_t> kernel_size,
              std::pair<int64_t,int64_t> stride, std::pair<int64_t,int64_t> padding,
              const std::string& mode) {
    if (input.ndim() != 4) throw std::invalid_argument("pygo::pool2d expects a 4D NCHW tensor");
    if (input.dtype() != DType::Float32) throw std::runtime_error("pygo::pool2d: only Float32 is implemented so far");
    bool is_max = (mode == "max");
    if (!is_max && mode != "avg") throw std::invalid_argument("pygo::pool2d: mode must be \"max\" or \"avg\"");

    auto [KH, KW] = kernel_size;
    auto [SH, SW] = stride;
    auto [PH, PW] = padding;

    const auto& dims = input.shape().dims;
    int64_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];
    int64_t H_out = (H + 2*PH - KH) / SH + 1;
    int64_t W_out = (W + 2*PW - KW) / SW + 1;
    if (H_out <= 0 || W_out <= 0) throw std::invalid_argument("pygo::pool2d: output size would be non-positive");

    Tensor in = input.contiguous();
    Tensor out = empty(Shape{N, C, H_out, W_out}, DType::Float32, input.device());
    const float* ip = in.data<float>();
    float* op = out.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* in_plane = ip + (n*C + c) * H * W;
            float* out_plane = op + (n*C + c) * H_out * W_out;
            for (int64_t oh = 0; oh < H_out; ++oh) {
                for (int64_t ow = 0; ow < W_out; ++ow) {
                    int64_t h_start = oh*SH - PH, w_start = ow*SW - PW;
                    float acc = is_max ? -std::numeric_limits<float>::infinity() : 0.0f;
                    int64_t count = 0;
                    for (int64_t kh = 0; kh < KH; ++kh) {
                        int64_t h = h_start + kh;
                        if (h < 0 || h >= H) continue;
                        for (int64_t kw = 0; kw < KW; ++kw) {
                            int64_t w = w_start + kw;
                            if (w < 0 || w >= W) continue;
                            float v = in_plane[h*W + w];
                            if (is_max) acc = std::max(acc, v);
                            else { acc += v; ++count; }
                        }
                    }
                    out_plane[oh*W_out + ow] = is_max ? acc : (count ? acc / (float)count : 0.0f);
                }
            }
        }
    }
    return out;
}

} // namespace pygo
