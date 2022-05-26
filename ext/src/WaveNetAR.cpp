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
    x_curr.resize(1u * WaveNet::getInputChannels());
    x_prev.resize(1u * WaveNet::getInputChannels());
    x_dist.resize(1u * WaveNet::getOutputChannels()); 
    this->reset();
}

void WaveNetAR::process(const float * input_data, float * const output_data, int total_samples)
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
        dist->sample(x_dist_data, x_curr_data, 1u);
        for (size_t c = 0; c < channels; c++)
        {
            output_data[t * channels + c] = x_curr[c] + input_data[t * channels + c];
            x_prev[c] = output_data[t  * channels + c];
        }
    }
}

void WaveNetAR::processConditional(const float * input_data, const float *conditioning,
                                   float *const output_data, int total_samples)
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
        // WaveNet::processConditional(x_prev_data, &conditioning[t * cond_channels], x_dist_data, 1u);
        WaveNet::processConditional(x_prev_data, cond.data(), x_dist_data, 1u);
        dist->sample(x_dist_data, x_curr_data, 1u);
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


void  WaveNetAR::setDistribution(std::string dist_name)
{
    // TODO: implement other distributions
    dist = std::unique_ptr<Distribution>(new GaussianDensity());
}

void WaveNetAR::setSamplingTemperature(float temperature)
{
    dist->setTemperature(temperature);
}