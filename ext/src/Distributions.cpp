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


void GaussianDensity::sample(const float * params, float * output, size_t timesteps)
{
    for (size_t t = 0; t < timesteps; t++)
    {
        const float m = params[2*t + 0u];
        const float log_s = params[2*t + 1u];
        unsigned int clamp = log_s < log_sigma_floor;
        const float s = exp((1u-clamp) * log_s + clamp * log_sigma_floor);
        const float z = distribution(generator);
        output[t] = m + s * z * temperature;
    }
}


} // distributions
} // glotnet