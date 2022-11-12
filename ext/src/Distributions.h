#ifndef GLOTNET_EXT_SRC_DISTRIBUTIONS_H_
#define GLOTNET_EXT_SRC_DISTRIBUTIONS_H_

#include <random>

namespace glotnet
{
namespace distributions
{

class Distribution
{
public:

    virtual ~Distribution() = default;
    virtual void sample(const float *params, float *output, size_t timesteps) = 0;
    virtual void setTemperature(float temperature) = 0;
};

class GaussianDensity : public Distribution
{
public:
    GaussianDensity(float log_sigma_floor=-7.0f, float temperature=1.0f);
    void sample(const float * params, float * output, size_t timesteps);
    void setTemperature(float t) {temperature = t;}; 

private:
    const float log_sigma_floor = -7.0f;
    float temperature = 1.0f;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution;

};

} // distributions
} // glotnet


#endif  // GLOTNET_EXT_SRC_DISTRIBUTIONS_H_

