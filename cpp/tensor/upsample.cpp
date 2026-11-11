#include "pygo/upsample.h"
#include <stdexcept>
#include <algorithm>
namespace pygo {

Tensor upsample(const Tensor& input, int64_t out_h, int64_t out_w, const std::string& mode) {
    if (input.ndim() != 4) throw std::invalid_argument("pygo::upsample expects a 4D NCHW tensor");
    if (input.dtype() != DType::Float32) throw std::runtime_error("pygo::upsample: only Float32 is implemented so far");
    bool nearest = (mode == "nearest");
    if (!nearest && mode != "bilinear") throw std::invalid_argument("pygo::upsample: mode must be \"nearest\" or \"bilinear\"");

    const auto& dims = input.shape().dims;
    int64_t N = dims[0], C = dims[1], H = dims[2], W = dims[3];
    Tensor in = input.contiguous();
    Tensor out = empty(Shape{N, C, out_h, out_w}, DType::Float32, input.device());
    const float* ip = in.data<float>();
    float* op = out.data<float>();

    float scale_h = (float)H / (float)out_h;
    float scale_w = (float)W / (float)out_w;

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            const float* in_plane = ip + (n*C + c) * H * W;
            float* out_plane = op + (n*C + c) * out_h * out_w;
            for (int64_t oh = 0; oh < out_h; ++oh) {
                for (int64_t ow = 0; ow < out_w; ++ow) {
                    if (nearest) {
                        int64_t ih = std::min(H - 1, (int64_t)(oh * scale_h));
                        int64_t iw = std::min(W - 1, (int64_t)(ow * scale_w));
                        out_plane[oh*out_w + ow] = in_plane[ih*W + iw];
                    } else {
                        // align_corners=false pixel-center mapping
                        float src_h = (oh + 0.5f) * scale_h - 0.5f;
                        float src_w = (ow + 0.5f) * scale_w - 0.5f;
                        src_h = std::max(0.0f, src_h);
                        src_w = std::max(0.0f, src_w);
                        int64_t h0 = (int64_t)src_h, w0 = (int64_t)src_w;
                        int64_t h1 = std::min(h0 + 1, H - 1);
                        int64_t w1 = std::min(w0 + 1, W - 1);
                        float dh = src_h - h0, dw = src_w - w0;
                        float top = in_plane[h0*W + w0]*(1-dw) + in_plane[h0*W + w1]*dw;
                        float bot = in_plane[h1*W + w0]*(1-dw) + in_plane[h1*W + w1]*dw;
                        out_plane[oh*out_w + ow] = top*(1-dh) + bot*dh;
                    }
                }
            }
        }
    }
    return out;
}

} // namespace pygo
