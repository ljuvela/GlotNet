#include "WaveNetAR.h"

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
        for (size_t j = 0; j < output_channels; j++)
        {
            // copy output to input
            input_buffer[j] = output_buffer[j];
            // copy to output buffer
            output_data[i * output_channels + j] = output_buffer[j];
        }
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
        for (size_t j = 0; j < output_channels; j++)
        {    
            // copy output to input
            input_buffer[j] = output_buffer[j];
            // copy to output buffer
            output_data[i * output_channels + j] = output_buffer[j];
        }
    }
}

