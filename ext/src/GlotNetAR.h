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

using namespace glotnet::distributions;

class GlotNetAR: public WaveNet
{
public:
    GlotNetAR(size_t input_channels, size_t output_channels,
            size_t convolution_channels, size_t skip_channels, 
            size_t cond_channels, size_t filter_width,
            std::string activation, std::vector<int> dilations,
            size_t lpc_order);
    void prepare();
    void flush(int64_t num_samples);
    void process(const float *input_data, const float *a_data, float *const output_data, int total_samples);
    void processConditional(const float *input_data, const float *a_data, const float *conditioning, float *const output_data, int total_samples);

    void setDistribution(std::string dist_name = "gaussian");
    void setSamplingTemperature(float temperature);

private:
    std::vector<float> x_dist;
    std::vector<float> x_buffer;
    std::vector<float> input_buffer;

    std::unique_ptr<Distribution> dist;
    const int64_t lpc_order;
    const int64_t input_channels;
    const int64_t output_channels;

};

}