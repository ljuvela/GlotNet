#include "glotnet/WaveNetAR.h"
#include "glotnet/Distributions.h"

#include <stdexcept>

namespace glotnet
{

using distributions::Distribution;
using distributions::GaussianDensity;

WaveNetAR::WaveNetAR(size_t input_channels, size_t output_channels,
                     size_t convolution_channels, size_t skip_channels, size_t cond_channels,
                     size_t filter_width, std::string activation, std::vector<int> dilations)
    : WaveNet(input_channels, output_channels,
              convolution_channels, skip_channels, cond_channels,
              filter_width, activation, dilations),
      input_channels(input_channels),
      output_channels(output_channels),
      dist(std::unique_ptr<Distribution>(new GaussianDensity()))
{
}

void WaveNetAR::prepare()
{
    WaveNet::prepare(1);
    x_curr.resize(1u * WaveNet::getInputChannels());
    x_prev.resize(1u * WaveNet::getInputChannels());
    x_dist.resize(1u * WaveNet::getOutputChannels());
    input_buffer.resize(input_channels);
    this->reset();
}

void WaveNetAR::flush(int64_t num_samples)
{
    this->prepare();
    float * const input_buffer_data = input_buffer.data();
    float * const x_dist_data = x_dist.data();
    std::fill(input_buffer.begin(), input_buffer.end(), 0.0f);
    for (int64_t t = 0; t < num_samples; t++)
    {
        WaveNet::process(input_buffer_data, x_dist_data, 1u);
    }
}

void WaveNetAR::process(
    const float *input_data,
    const float *temperature,
    float *const output_data,
    int total_samples
)
{
    this->prepare();
    const int dist_channels = WaveNet::getOutputChannels();
    const int channels = WaveNet::getInputChannels();

    float * const x_prev_data = x_prev.data();
    float * const x_curr_data = x_curr.data();
    float * const x_dist_data = x_dist.data();
    std::fill(x_prev.begin(), x_prev.end(), 0.0f);
    std::fill(x_curr.begin(), x_curr.end(), 0.0f);
    std::fill(x_dist.begin(), x_dist.end(), 0.0f);

    // Timesteps 1:total_samples
    for (int64_t t = 0; t < total_samples; t++)
    {
        WaveNet::process(x_prev_data, x_dist_data, 1u);
        dist->sample(x_dist_data, x_curr_data, 1u, &temperature[t]);
        for (size_t c = 0; c < channels; c++)
        {
            output_data[t * channels + c] = x_curr[c] + input_data[t * channels + c];
            x_prev[c] = output_data[t  * channels + c];
        }
    }
}

void WaveNetAR::processConditional(
    const float *input_data,
    const float *conditioning,
    const float *temperature,
    float *const output_data,
    int total_samples)
{
    this->prepare();
    const int dist_channels = WaveNet::getOutputChannels();
    const int channels = WaveNet::getInputChannels();
    const int cond_channels = WaveNet::getCondChannels();

    float * const x_prev_data = x_prev.data();
    float * const x_curr_data = x_curr.data();
    float * const x_dist_data = x_dist.data();
    std::fill(x_prev.begin(), x_prev.end(), 0.0f);
    std::fill(x_curr.begin(), x_curr.end(), 0.0f);
    std::fill(x_dist.begin(), x_dist.end(), 0.0f);

    std::vector<float> cond(cond_channels);
    std::fill(cond.begin(), cond.end(), 0.0f);

    // Timesteps 1:total_samples
    for (int64_t t = 0; t < total_samples; t++)
    {
        WaveNet::processConditional(x_prev_data, cond.data(), x_dist_data, 1u);
        dist->sample(x_dist_data, x_curr_data, 1u, &temperature[t]);
        for (size_t c = 0; c < channels; c++)
        {
            output_data[t * channels + c] = x_curr[c] + input_data[t * channels + c];
            x_prev[c] = output_data[t  * channels + c];
        }
        for (size_t c = 0; c < cond_channels; c++)
        {
            cond[c] = conditioning[t * cond_channels + c]; // TODO: make conditioning one sample earlier
        }

    }
}

void WaveNetAR::setDistribution(const std::string &dist_name, int num_bits)
{
    distributions::DistributionSpec spec;
    spec.name = dist_name;
    spec.num_bits = num_bits;
    setDistribution(spec);
}

void WaveNetAR::setDistribution(const distributions::DistributionSpec &spec)
{
    auto new_dist = distributions::create(spec);
    if (new_dist->paramsDim() != static_cast<size_t>(output_channels))
        throw std::invalid_argument(
            "WaveNetAR: distribution '" + spec.name + "' needs "
            + std::to_string(new_dist->paramsDim()) + " output channels, model has "
            + std::to_string(output_channels));
    dist = std::move(new_dist);
}

void WaveNetAR::setSeed(uint64_t seed)
{
    dist->seed(seed);
}

} // namespace glotnet
