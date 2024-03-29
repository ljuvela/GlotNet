#include "GlotNetAR.h"
#include "Distributions.h"

GlotNetAR::GlotNetAR(size_t input_channels, size_t output_channels,
                     size_t convolution_channels, size_t skip_channels, size_t cond_channels,
                     size_t filter_width, std::string activation, std::vector<int> dilations,
                     size_t lpc_order, bool sample_after_filtering)
    : WaveNet(input_channels, output_channels,
              convolution_channels, skip_channels, cond_channels,
              filter_width, activation, dilations),
      dist(std::unique_ptr<Distribution>(new GaussianDensity())),
      lpc_order(lpc_order),
      input_channels(input_channels),
      output_channels(output_channels),
      sample_after_filtering(sample_after_filtering)
{
}

void GlotNetAR::prepare()
{
    // prepare wavenet for single timestep prediction
    WaveNet::prepare(1u);
    x_dist.resize(output_channels);
    x_buffer.resize(lpc_order);
    input_buffer.resize(input_channels);
    this->reset();
}

void GlotNetAR::flush(int64_t num_samples)
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

void GlotNetAR::process(const float *input_data,
                        const float *a_data,
                        const float *temperature,
                        float *const output_data,
                        int total_samples
                     )
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

        // get distribution from wavenet
        WaveNet::process(input_buffer_data, x_dist_data, 1u);

        // get current prediction from input
        float p_curr = input_buffer[1];
        float x_curr;
        float e_curr;

        if (sample_after_filtering)
        {
            x_dist[0] += p_curr;
            // sample from distribution
            dist->sample(x_dist_data, &x_curr, 1u, &temperature[t]);
            e_curr = x_curr - p_curr;
        }
        else
        {
            dist->sample(x_dist_data, &e_curr, 1u, &temperature[t]);
            // compute current sample
            x_curr = e_curr + p_curr;
        }

        // Update signal buffer
        for (size_t i = lpc_order-1; i > 0; i--)
        {
            x_buffer[i] = x_buffer[i-1];
        }
        x_buffer[0] = x_curr;

        // Update prediction
        float p_next = 0.0;
        for (size_t i = 0; i < lpc_order; i++)
        {
            const float a = a_data[t * (lpc_order+1) + (i+1)];
            p_next -=  a * x_buffer[i];
        }

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
                                   const float *temperature,
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

        if (false)
        {
            // set distribution to white noise
            x_dist[0] = 0.0f; // mean
            x_dist[1] = -6.0f; // log variance

            // x_dist[0] = input_data[t];
            // x_dist[1] = -14.0f;
        }
        else
        {
            // get distribution from wavenet
            WaveNet::processConditional(input_buffer_data, cond.data(), x_dist_data, 1u);
        }

        dist->sample(x_dist_data, &e_curr, 1u, &temperature[t]);

        // get current prediction from input
        float p_curr = input_buffer[1];

        // compute current sample
        float x_curr = e_curr + p_curr;

        // Update signal buffer
        for (size_t i = lpc_order-1; i > 0; i--)
        {
            x_buffer[i] = x_buffer[i-1];
        }
        x_buffer[0] = x_curr;

        // Update prediction
        float p_next = 0.0;
        for (size_t i = 0; i < lpc_order; i++)
        {
            const float a = a_data[t * (lpc_order+1) + (i+1)];
            p_next -=  a * x_buffer[i];     
        }

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
