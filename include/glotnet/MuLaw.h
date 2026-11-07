#pragma once

// Mu-law companding and quantization (ITU-T G.711), header-only.
// Same conventions as glotnet/sigproc/mulaw.py: signal in [-1, 1],
// mu = 2^bits - 1, class indices in [0, mu].

#include <algorithm>
#include <cmath>

namespace glotnet
{
namespace mulaw
{

inline int muFromBits(int num_bits) { return (1 << num_bits) - 1; }

// [-1, 1] -> [-1, 1]; input is clipped to [-1, 1]
inline float compress(float x, float mu)
{
    x = std::min(1.0f, std::max(-1.0f, x));
    return std::copysign(std::log1p(mu * std::abs(x)) / std::log1p(mu), x);
}

inline float expand(float y, float mu)
{
    return std::copysign(std::expm1(std::abs(y) * std::log1p(mu)) / mu, y);
}

// companded value -> class index in [0, mu]
inline int quantize(float y, int mu)
{
    const int q = static_cast<int>(std::floor((y + 1.0f) * 0.5f * mu + 0.5f));
    return std::min(mu, std::max(0, q));
}

// class index -> companded value
inline float dequantize(int q, int mu)
{
    return 2.0f * static_cast<float>(q) / static_cast<float>(mu) - 1.0f;
}

inline int encode(float x, int mu) { return quantize(compress(x, static_cast<float>(mu)), mu); }
inline float decode(int q, int mu) { return expand(dequantize(q, mu), static_cast<float>(mu)); }

} // namespace mulaw
} // namespace glotnet
