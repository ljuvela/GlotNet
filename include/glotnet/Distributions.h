#ifndef GLOTNET_EXT_SRC_DISTRIBUTIONS_H_
#define GLOTNET_EXT_SRC_DISTRIBUTIONS_H_

#include <random>
#include <time.h>

namespace glotnet
{
namespace distributions
{

class Distribution
{
public:
    virtual ~Distribution() = default;
    virtual void sample(const float *params, float *output, size_t timesteps, const float *temperature) = 0;
};

class GaussianDensity : public Distribution
{
public:
    GaussianDensity(float log_sigma_floor = -14.0f);
    void sample(const float *params, float *output, size_t timesteps, const float *temperature);
    void setVarianceShape(float shape);

private:
    const float log_sigma_floor = -14.0f;
    float variance_shape = 0.0f;
    float one_per_log_one_plus_mu = 0.0f;
    unsigned int seed = time(0);
    std::default_random_engine generator{seed};
    std::normal_distribution<float> distribution;

    float shapeVariance(float sigma);

};

} // distributions
} // glotnet


#endif  // GLOTNET_EXT_SRC_DISTRIBUTIONS_H_

