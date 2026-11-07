
#include <algorithm>
#include <cmath>
#include <random>
#include <stdexcept>
#include "glotnet/Distributions.h"
#include "glotnet/MuLaw.h"

namespace glotnet
{
namespace distributions
{

Distribution::Distribution()
    : generator(std::random_device{}())
{
}

void Distribution::seed(uint64_t value)
{
    generator.seed(value);
}

GaussianDensity::GaussianDensity(float log_sigma_floor)
    : log_sigma_floor(log_sigma_floor),
    distribution(0.0f, 1.0f)
{

}

void GaussianDensity::seed(uint64_t value)
{
    Distribution::seed(value);
    distribution.reset(); // drop cached normal deviate
}

void GaussianDensity::sample(const float * params, float * output, size_t timesteps, const float * temperature)
{
    for (size_t t = 0; t < timesteps; t++)
    {
        const float m = params[2*t + 0u];
        const float log_s = params[2*t + 1u];
        const unsigned int clamp = log_s < log_sigma_floor;
        const float s = exp((1u-clamp) * log_s + clamp * log_sigma_floor);
        const float z = distribution(generator);
        output[t] = m + s * z * temperature[t];
    }
}

void GaussianDensity::setVarianceShape(float shape)
{
    variance_shape = shape;
    one_per_log_one_plus_mu = std::log(1 + variance_shape);
}

float GaussianDensity::shapeVariance(float sigma)
{
    const float val = std::log(1.f + variance_shape * std::abs(sigma)) * one_per_log_one_plus_mu;
    // clamp value to 1.0
    const float val_clamped = std::min(val, 1.0f);
    return std::copysignf(val_clamped, sigma);
}

CategoricalDensity::CategoricalDensity(int num_bits)
    : num_bits(num_bits),
      num_classes(num_bits >= 1 && num_bits <= 16 ? size_t(1) << num_bits : 0),
      mu(mulaw::muFromBits(num_bits)),
      cdf(num_classes)
{
    if (num_classes == 0)
        throw std::invalid_argument("CategoricalDensity: num_bits must be in [1, 16], got "
                                    + std::to_string(num_bits));
}

int CategoricalDensity::sampleIndex(const float *logits, float temperature)
{
    const float *max_it = std::max_element(logits, logits + num_classes);
    if (!(temperature > 0.0f))
        return static_cast<int>(max_it - logits);

    // unnormalized CDF of softmax(logits / T), shifted by the max for stability
    const double max_logit = *max_it;
    const double inv_temp = 1.0 / temperature;
    double total = 0.0;
    for (size_t k = 0; k < num_classes; k++)
    {
        total += std::exp((logits[k] - max_logit) * inv_temp);
        cdf[k] = total;
    }
    // inverse CDF: first class with cdf > u
    const double u = uniform(generator) * total;
    const auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
    const size_t k = std::min(static_cast<size_t>(it - cdf.begin()), num_classes - 1);
    return static_cast<int>(k);
}

void CategoricalDensity::sample(const float *params, float *output, size_t timesteps, const float *temperature)
{
    for (size_t t = 0; t < timesteps; t++)
    {
        const int q = sampleIndex(params + t * num_classes, temperature[t]);
        output[t] = mulaw::decode(q, mu);
    }
}

std::unique_ptr<Distribution> create(const std::string &name, int num_bits)
{
    if (name == "gaussian")
        return std::make_unique<GaussianDensity>();
    if (name == "categorical")
        return std::make_unique<CategoricalDensity>(num_bits);
    throw std::invalid_argument("Unknown distribution '" + name
                                + "' (expected 'gaussian' or 'categorical')");
}

} // distributions
} // glotnet
