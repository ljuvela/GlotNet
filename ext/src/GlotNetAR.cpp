#include "GlotNetAR.h"
#include "Distributions.h"

GlotNetAR::GlotNetAR(size_t input_channels, size_t output_channels,
                     size_t convolution_channels, size_t skip_channels, size_t cond_channels,
                     size_t filter_width, std::string activation, std::vector<int> dilations,
                     size_t lpc_order)
    : WaveNet(input_channels, output_channels,
              convolution_channels, skip_channels, cond_channels,
              filter_width, activation, dilations),
      dist(std::unique_ptr<Distribution>(new GaussianDensity())),
      lpc_order(lpc_order),
      input_channels(input_channels),
      output_channels(output_channels)
{
}

void GlotNetAR::prepare()
{
    // prepare wavenet for single timestep prediction
    WaveNet::prepare(1u);
    x_dist.resize(output_channels);
    x_buffer.resize( lpc_order);
    input_buffer.resize(input_channels);
    this->reset();
}

void GlotNetAR::process(const float * input_data, const float * a_data, float * const output_data, int total_samples)
{
    // a_data is the LPC coefficients, shape (total_samples, lpc_order+1)


    this->prepare();
    const int dist_channels = WaveNet::getOutputChannels();
    const int channels = WaveNet::getInputChannels();

    float * const input_buffer_data = input_buffer.data();
    float * const x_dist_data = x_dist.data();

    std::fill(x_buffer.begin(), x_buffer.end(), 0.0f);
    std::fill(input_buffer.begin(), input_buffer.end(), 0.0f);

    // Timesteps 1:total_samples
    for (int64_t t = 0; t < total_samples; t++)
    {
        float e_curr;
        WaveNet::process(input_buffer_data, x_dist_data, 1u);
        dist->sample(x_dist_data, &e_curr, 1u);

        // get current prediction from input
        float p_curr = input_buffer[1];

        // compute current sample
        float x_curr = e_curr + p_curr;

        // Update prediction
        float p_next = 0.0;
        for (size_t i = 0; i < lpc_order; i++)
        {
            float a = a_data[t * (lpc_order+1) + (i+1)];
            std::cout << "a:" << a << std::endl;
            std::cout << "x:" << x_buffer[i] << std::endl; 
            p_next -=  a * x_buffer[i];
           
        }
        // Roll signal buffer
        for (size_t i = lpc_order-1; i > 0; i--)
        {
            x_buffer[i] = x_buffer[i-1];
        }
        x_buffer[0] = x_curr;

        // Update input buffer
        input_buffer[0] = e_curr;
        input_buffer[1] = p_next;
        input_buffer[2] = x_curr;

        // Copy to output
        output_data[t] = x_curr;
    }
}

void GlotNetAR::processConditional(const float *input_data,
                                   const float *a_data,
                                   const float *conditioning,
                                   float *const output_data,
                                   int total_samples)
{
    this->prepare();
    const int dist_channels = WaveNet::getOutputChannels();
    const int channels = WaveNet::getInputChannels();
    const int cond_channels = WaveNet::getCondChannels();

    float * const input_buffer_data = input_buffer.data();
    float * const x_dist_data = x_dist.data();

    std::fill(x_buffer.begin(), x_buffer.end(), 0.0f);
    std::fill(input_buffer.begin(), input_buffer.end(), 0.0f);
    
    std::vector<float> cond(cond_channels);
    std::fill(cond.begin(), cond.end(), 0.0f);

    // Timesteps 1:total_samples
    for (int64_t t = 0; t < total_samples; t++)
    {
        float e_curr;
        WaveNet::processConditional(input_buffer_data, cond.data(), x_dist_data, 1u);
        dist->sample(x_dist_data, &e_curr, 1u);

        // get current prediction from input
        float p_curr = input_buffer[1];

        // compute current sample
        float x_curr = e_curr + p_curr;

        // Update prediction
        float p_next = 0.0;
        for (size_t i = 0; i < lpc_order; i++)
        {
            p_next -= a_data[t * (lpc_order+1) + (i+1)] * x_buffer[i];
        }
        // Roll signal buffer
        for (size_t i = lpc_order-1; i > 0; i--)
        {
            x_buffer[i] = x_buffer[i-1];
        }
        x_buffer[0] = x_curr;

        // Update input buffer
        input_buffer[0] = e_curr;
        input_buffer[1] = p_next;
        input_buffer[2] = x_curr;

        // Copy to output
        output_data[t] = x_curr;

        // Update conditioning
        for (size_t c = 0; c < cond_channels; c++)
        {
            cond[c] = conditioning[t * cond_channels + c]; // TODO: make conditioning one sample earlier
        }

    }
}

void  GlotNetAR::setDistribution(std::string dist_name)
{
    // TODO: implement other distributions
    dist = std::unique_ptr<Distribution>(new GaussianDensity());
}

void GlotNetAR::setSamplingTemperature(float temperature)
{
    dist->setTemperature(temperature);
}