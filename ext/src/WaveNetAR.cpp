#include "WaveNetAR.h"
#include "Distributions.h"

WaveNetAR::WaveNetAR(size_t input_channels, size_t output_channels,
                     size_t convolution_channels, size_t skip_channels, size_t cond_channels,
                     size_t filter_width, std::string activation, std::vector<int> dilations)
    : WaveNet(input_channels, output_channels,
              convolution_channels, skip_channels, cond_channels,
              filter_width, activation, dilations),
      dist(std::unique_ptr<Distribution>(new GaussianDensity()))
{
}

void WaveNetAR::prepare()
{
    WaveNet::prepare(1);
    input_buffer.resize(1u * WaveNet::getInputChannels()); // only single timestep
    output_buffer.resize(1u * WaveNet::getOutputChannels()); // only single timestep
    this->reset();
}


void WaveNetAR::process(float * const output_data, int total_samples)
{
    std::fill(input_buffer.begin(), input_buffer.end(), 0.0f);
    const int output_channels = WaveNet::getOutputChannels();
    for (int i = 0; i < total_samples; i++)
    {
        WaveNet::process(input_buffer.data(), output_buffer.data(), 1u);

        float x = -1;
        dist->sample(output_buffer.data(), &x, 1u);
        input_buffer[0] = x;
        output_data[i] = x;
    }
}

void WaveNetAR::processConditional(const float *conditioning,
                                   float *const output_data, int total_samples)
{
    std::fill(input_buffer.begin(), input_buffer.end(), 0.0f);
    const int output_channels = WaveNet::getOutputChannels();
    const int cond_channels = WaveNet::getCondChannels();
    for (int i = 0; i < total_samples; i++)
    {
        WaveNet::processConditional(input_buffer.data(), &conditioning[i * cond_channels], output_buffer.data(), 1u);

        float x = -1;
        dist->sample(output_buffer.data(), &x, 1u);
        input_buffer[0] = x;
        output_data[i] = x;
    }
}


void  WaveNetAR::setDistribution(std::string dist_name)
{
    dist = std::unique_ptr<Distribution>(new GaussianDensity());
}

void WaveNetAR::setSamplingTemperature(float temperature)
{
    dist->setTemperature(temperature);
}