#ifndef GLOTNET_EXT_SRC_DISTRIBUTIONS_H_
#define GLOTNET_EXT_SRC_DISTRIBUTIONS_H_

#include <random>

namespace glotnet
{
namespace distributions
{

class GaussianDensity
{
public:
    GaussianDensity(float log_sigma_floor=-7.0f, float temperature=1.0f);
    void sample(const float * params, float * output, size_t timesteps);

private:
    const float log_sigma_floor = -7.0f;
    float temperature = 1.0f;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution;

};

} // distributions
} // glotnet


#endif  // GLOTNET_EXT_SRC_DISTRIBUTIONS_H_

