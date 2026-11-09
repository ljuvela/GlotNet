
#include <algorithm>
#include <cmath>
#include <limits>
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

void Distribution::shiftLocation(float *, float) const
{
    throw std::logic_error("this distribution has no location parameter to shift");
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

Family familyFromName(const std::string &name)
{
    if (name == "gaussian")
        return Family::Gaussian;
    if (name == "laplace")
        return Family::Laplace;
    if (name == "logistic")
        return Family::Logistic;
    throw std::invalid_argument("Unknown location-scale family '" + name
                                + "' (expected 'gaussian', 'laplace' or 'logistic')");
}

LocationScaleMixture::LocationScaleMixture(std::vector<Family> components_, bool shared_loc,
                                           int discretize_bits, float log_scale_floor)
    : components(std::move(components_)),
      num_components(components.size()),
      shared_loc(shared_loc),
      discretize_bits(discretize_bits),
      log_scale_floor(log_scale_floor),
      params_dim(shared_loc ? 2 * num_components + 1 : 3 * num_components),
      cdf(num_components)
{
    if (num_components == 0)
        throw std::invalid_argument("LocationScaleMixture needs at least one component");
    if (discretize_bits < 0 || discretize_bits > 24)
        throw std::invalid_argument("LocationScaleMixture: discretize_bits must be in [0, 24] "
                                    "(0 = continuous), got " + std::to_string(discretize_bits));
}

void LocationScaleMixture::seed(uint64_t value)
{
    Distribution::seed(value);
    normal.reset(); // drop cached normal deviate
}

void LocationScaleMixture::shiftLocation(float *params, float offset) const
{
    const size_t num_locs = shared_loc ? 1 : num_components;
    for (size_t k = 0; k < num_locs; k++)
        params[num_components + k] += offset;
}

double LocationScaleMixture::uniformOpen()
{
    // uniform() is in [0, 1): keep away from both ends (inverse CDFs diverge)
    const double u = uniform(generator);
    return std::min(std::max(u, std::numeric_limits<double>::min()),
                    1.0 - std::numeric_limits<double>::epsilon());
}

float LocationScaleMixture::sampleStandard(Family family)
{
    switch (family)
    {
    case Family::Gaussian:
        return normal(generator);
    case Family::Laplace:
    {
        const double v = uniformOpen() - 0.5;
        return static_cast<float>(-std::copysign(1.0, v) * std::log1p(-2.0 * std::abs(v)));
    }
    case Family::Logistic:
    {
        const double u = uniformOpen();
        return static_cast<float>(std::log(u) - std::log1p(-u));
    }
    }
    return 0.0f;
}

size_t LocationScaleMixture::sampleComponent(const float *params)
{
    // inverse CDF of softmax(logits), shifted by the max for stability
    const float *logits = params;
    const double max_logit = *std::max_element(logits, logits + num_components);
    double total = 0.0;
    for (size_t k = 0; k < num_components; k++)
    {
        total += std::exp(logits[k] - max_logit);
        cdf[k] = total;
    }
    const double u = uniform(generator) * total;
    const auto it = std::upper_bound(cdf.begin(), cdf.end(), u);
    return std::min(static_cast<size_t>(it - cdf.begin()), num_components - 1);
}

float LocationScaleMixture::sampleOne(const float *params, float temperature)
{
    const float *logits = params;
    const size_t k = temperature > 0.0f
        ? sampleComponent(params)
        : static_cast<size_t>(std::max_element(logits, logits + num_components) - logits);

    const float loc = params[num_components + (shared_loc ? 0 : k)];
    const float log_scale = std::max(params[(shared_loc ? num_components + 1 : 2 * num_components) + k],
                                     log_scale_floor);
    float x = loc;
    if (temperature > 0.0f)
        x += temperature * std::exp(log_scale) * sampleStandard(components[k]);

    if (discretize_bits > 0)
    {
        // clip to [-1, 1] and snap to 2^bits levels; nearbyint rounds half to
        // even like torch.round
        const double n = static_cast<double>((uint64_t(1) << discretize_bits) - 1);
        const double xc = std::min(std::max(static_cast<double>(x), -1.0), 1.0);
        x = static_cast<float>(std::nearbyint((xc + 1.0) * (n / 2.0)) * (2.0 / n) - 1.0);
    }
    return x;
}

void LocationScaleMixture::sample(const float *params, float *output, size_t timesteps, const float *temperature)
{
    for (size_t t = 0; t < timesteps; t++)
        output[t] = sampleOne(params + t * params_dim, temperature[t]);
}

std::unique_ptr<Distribution> create(const DistributionSpec &spec)
{
    if (spec.name == "gaussian")
        return std::make_unique<GaussianDensity>();
    if (spec.name == "categorical")
        return std::make_unique<CategoricalDensity>(spec.num_bits);
    if (spec.name == "mixture")
    {
        std::vector<Family> families;
        for (const auto &c : spec.components)
            families.push_back(familyFromName(c));
        return std::make_unique<LocationScaleMixture>(families, spec.shared_loc,
                                                      spec.discretize_bits);
    }
    throw std::invalid_argument("Unknown distribution '" + spec.name
                                + "' (expected 'gaussian', 'categorical' or 'mixture')");
}

std::unique_ptr<Distribution> create(const std::string &name, int num_bits)
{
    DistributionSpec spec;
    spec.name = name;
    spec.num_bits = num_bits;
    return create(spec);
}

} // distributions
} // glotnet
