
#include <cmath>
#include <random>
#include "Distributions.h"

namespace glotnet
{
namespace distributions
{

GaussianDensity::GaussianDensity(float log_sigma_floor, float temperature)
    : log_sigma_floor(log_sigma_floor),
    temperature(temperature),
    distribution(0.0f, 1.0f)
{

}

void GaussianDensity::prepare(size_t timesteps_new)
{
    timesteps = timesteps_new;
}

void GaussianDensity::sample(const float * mean, const float * log_sigma, float * output)
{
    for (size_t t = 0; t < timesteps; t++)
    {
        const float m     = mean[t];
        const float log_s = log_sigma[t];
        unsigned int clamp = log_s < log_sigma_floor;
        const float s = expf((1u-clamp) * log_s + clamp * log_sigma_floor);
        const float z = distribution(generator);
        output[t] = m + s * z * temperature;
    }
}


} // distributions
} // glotnet