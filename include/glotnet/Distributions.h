#ifndef GLOTNET_EXT_SRC_DISTRIBUTIONS_H_
#define GLOTNET_EXT_SRC_DISTRIBUTIONS_H_

#include <cstdint>
#include <memory>
#include <random>
#include <string>
#include <vector>

namespace glotnet
{
namespace distributions
{

class Distribution
{
public:
    Distribution();
    virtual ~Distribution() = default;

    // Number of parameters per timestep (network output channels)
    virtual size_t paramsDim() const = 0;

    // params: (timesteps, paramsDim) row-major; output, temperature: (timesteps)
    virtual void sample(const float *params, float *output, size_t timesteps, const float *temperature) = 0;

    // Reseed the random number generator (default: seeded from std::random_device)
    virtual void seed(uint64_t value);

protected:
    std::mt19937_64 generator;
};

class GaussianDensity : public Distribution
{
public:
    GaussianDensity(float log_sigma_floor = -14.0f);
    size_t paramsDim() const override { return 2; }
    void sample(const float *params, float *output, size_t timesteps, const float *temperature) override;
    void seed(uint64_t value) override;
    void setVarianceShape(float shape);

private:
    const float log_sigma_floor = -14.0f;
    float variance_shape = 0.0f;
    float one_per_log_one_plus_mu = 0.0f;
    std::normal_distribution<float> distribution;

    float shapeVariance(float sigma);

};

// Categorical distribution over mu-law quantized values (2^num_bits classes).
// params are logits; samples are returned as signal values in [-1, 1]
// (dequantized and expanded). Logits are divided by the temperature;
// temperature <= 0 returns the most likely value.
class CategoricalDensity : public Distribution
{
public:
    explicit CategoricalDensity(int num_bits = 8);
    size_t paramsDim() const override { return num_classes; }
    void sample(const float *params, float *output, size_t timesteps, const float *temperature) override;
    int sampleIndex(const float *logits, float temperature);

private:
    const int num_bits;
    const size_t num_classes;
    const int mu;
    std::vector<double> cdf; // preallocated, no allocation while sampling
    std::uniform_real_distribution<double> uniform{0.0, 1.0};
};

// Factory: "gaussian" or "categorical" (num_bits used by categorical only).
// Throws std::invalid_argument for unknown names.
std::unique_ptr<Distribution> create(const std::string &name, int num_bits = 8);

} // distributions
} // glotnet


#endif  // GLOTNET_EXT_SRC_DISTRIBUTIONS_H_
