#include "pygo/stft.h"
#include "pygo/fft.h"
#include <stdexcept>
#include <cmath>
#include <vector>

namespace pygo {

std::pair<Tensor, Tensor> stft(const Tensor& input, int64_t n_fft, int64_t hop_length) {
    if (input.ndim() != 1) throw std::invalid_argument("pygo::stft currently supports a 1D input signal only");
    if (input.dtype() != DType::Float32) throw std::runtime_error("pygo::stft: only Float32 is implemented so far");
    if (hop_length <= 0) throw std::invalid_argument("pygo::stft: hop_length must be positive");

    int64_t T = input.numel();
    if (T < n_fft) throw std::invalid_argument("pygo::stft: signal is shorter than n_fft");
    int64_t n_frames = 1 + (T - n_fft) / hop_length;

    Tensor sig = input.contiguous();
    const float* sp = sig.data<float>();

    // Periodic Hann window (matches torch.stft's default `window=None` ->
    // rectangular is actually torch's real default, but an explicit Hann
    // window is far more common in practice and avoids severe spectral
    // leakage, so it's applied here rather than left rectangular).
    std::vector<float> window(n_fft);
    for (int64_t i = 0; i < n_fft; ++i)
        window[i] = 0.5f - 0.5f * std::cos(2.0f * (float)M_PI * (float)i / (float)n_fft);

    Tensor frames = empty(Shape{n_frames, n_fft}, DType::Float32, input.device());
    float* fp = frames.data<float>();
    for (int64_t f = 0; f < n_frames; ++f) {
        int64_t start = f * hop_length;
        for (int64_t i = 0; i < n_fft; ++i)
            fp[f * n_fft + i] = sp[start + i] * window[i];
    }

    return fft(frames, /*dim=*/-1);
}

} // namespace pygo
