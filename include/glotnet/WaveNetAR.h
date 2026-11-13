#pragma once

#include <string>
#include "Activations.h"
#include "ConvolutionStack.h"
#include "WaveNet.h"
#include "Distributions.h"

#include <string>
#include <memory>

namespace glotnet
{

class WaveNetAR: public WaveNet
{
public:
    WaveNetAR(size_t input_channels, size_t output_channels,
            size_t convolution_channels, size_t skip_channels, size_t cond_channels,
            size_t filter_width, std::string activation, std::vector<int> dilations);
    void prepare();
    void flush(int64_t num_samples);
    void process(const float *input_data,
                 const float *temperature,
                 float *const output_data,
                 int total_samples);
    void processConditional(const float *input_data,
                            const float *conditioning,
                            const float *temperature,
                            float *const output_data,
                            int total_samples);

    // "gaussian", "categorical" or "mixture" (see distributions::DistributionSpec).
    // Throws std::invalid_argument if the distribution's paramsDim() does
    // not match the output channels.
    void setDistribution(const distributions::DistributionSpec &spec);
    void setDistribution(const std::string &dist_name = "gaussian", int num_bits = 8);
    // Reseed the sampler (call after setDistribution)
    void setSeed(uint64_t seed);
    void setSamplingTemperature(float temperature);

private:
    std::vector<float> x_curr;
    std::vector<float> x_prev;
    std::vector<float> x_dist;
    std::vector<float> input_buffer;
    std::unique_ptr<distributions::Distribution> dist;
    const int64_t input_channels;
    const int64_t output_channels;

};

}