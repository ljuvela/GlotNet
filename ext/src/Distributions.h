#ifndef GLOTNET_EXT_SRC_DISTRIBUTIONS_H_
#define GLOTNET_EXT_SRC_DISTRIBUTIONS_H_

#include <cstdlib>

namespace glotnet
{
namespace distributions
{

class GaussianDensity
{
public:
    GaussianDensity(float log_sigma_floor, float temperature=1.0f);
    ~GaussianDensity();
    void prepare(size_t timesteps_new);
    void sample(const float *m, const float *log_s, float *output);

private:
    size_t timesteps;
    const float log_sigma_floor = -7.0f;
    float temperature = 1.0f;
    std::default_random_engine generator;
    std::normal_distribution<float> distribution;

};

} // distributions
} // glotnet


#endif  // GLOTNET_EXT_SRC_DISTRIBUTIONS_H_

