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

    // Location-scale distributions can describe x + offset by moving their
    // location parameters (GlotNet sample_after_filtering). Others cannot.
    virtual bool canShiftLocation() const { return false; }
    // params: one timestep (paramsDim()); throws std::logic_error unless canShiftLocation()
    virtual void shiftLocation(float *params, float offset) const;

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
    bool canShiftLocation() const override { return true; }
    void shiftLocation(float *params, float offset) const override { params[0] += offset; }
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

// Location-scale families of the mixture components
enum class Family
{
    Gaussian,  // scale = standard deviation
    Laplace,   // density exp(-|x - loc| / b) / (2 b)
    Logistic   // CDF sigmoid((x - loc) / s)
};

// "gaussian", "laplace" or "logistic"; throws std::invalid_argument otherwise
Family familyFromName(const std::string &name);

// Mixture of location-scale densities, matching the Python
// glotnet.losses.distributions.LocationScaleMixture. Parameters per timestep
// (K components):
//   shared_loc = false: [logits (K), locs (K), log_scales (K)]  -> 3K
//   shared_loc = true:  [logits (K), loc (1),  log_scales (K)]  -> 2K + 1
// Sampling: a component from softmax(logits) (not tempered), then
// loc + temperature * scale * z with z from the component's family;
// temperature <= 0 returns the location of the most probable component.
// discretize_bits > 0 clips samples to [-1, 1] and snaps them to the
// 2^bits levels. Log scales are floored at log_scale_floor, as in GaussianDensity.
class LocationScaleMixture : public Distribution
{
public:
    LocationScaleMixture(std::vector<Family> components, bool shared_loc = false,
                         int discretize_bits = 0, float log_scale_floor = -14.0f);
    size_t paramsDim() const override { return params_dim; }
    void sample(const float *params, float *output, size_t timesteps, const float *temperature) override;
    void seed(uint64_t value) override;
    bool canShiftLocation() const override { return true; }
    void shiftLocation(float *params, float offset) const override;

    // One timestep: the sampled component index (exposed for tests)
    size_t sampleComponent(const float *params);
    // One timestep: the sampled value
    float sampleOne(const float *params, float temperature);
    // A standard (location 0, scale 1) draw from a family
    float sampleStandard(Family family);

    size_t numComponents() const { return num_components; }

private:
    const std::vector<Family> components;
    const size_t num_components;
    const bool shared_loc;
    const int discretize_bits;
    const float log_scale_floor;
    const size_t params_dim;
    std::vector<double> cdf; // preallocated, no allocation while sampling
    std::uniform_real_distribution<double> uniform{0.0, 1.0};
    std::normal_distribution<float> normal{0.0f, 1.0f};

    double uniformOpen(); // in (0, 1)
};

// What to create: name "gaussian", "categorical" or "mixture".
// num_bits: mu-law bits (categorical). components, shared_loc and
// discretize_bits (0 = continuous): mixture only.
struct DistributionSpec
{
    std::string name = "gaussian";
    int num_bits = 8;
    std::vector<std::string> components;
    bool shared_loc = false;
    int discretize_bits = 0;
};

// Factory. Throws std::invalid_argument for unknown names or invalid settings.
std::unique_ptr<Distribution> create(const DistributionSpec &spec);
std::unique_ptr<Distribution> create(const std::string &name, int num_bits = 8);

} // distributions
} // glotnet


#endif  // GLOTNET_EXT_SRC_DISTRIBUTIONS_H_
