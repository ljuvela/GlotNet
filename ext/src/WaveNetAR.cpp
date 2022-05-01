#include "WaveNetAR.h"

WaveNetAR::WaveNetAR(size_t input_channels, size_t output_channels,
                     size_t convolution_channels, size_t skip_channels, size_t cond_channels,
                     size_t filter_width, std::string activation, std::vector<int> dilations)
    : input_channels(input_channels),
      output_channels(output_channels),
      filter_width(filter_width),
      skip_channels(skip_channels * (int)dilations.size()),
      conv_stack(convolution_channels, skip_channels, cond_channels, filter_width, dilations, activation),
      input_layer(input_channels, convolution_channels, 0, 1, 1, false, "tanh"),
      output_layer1(convolution_channels * dilations.size(), convolution_channels, 0, 1, 1, false, "tanh"),
      output_layer2(convolution_channels, output_channels, 0, 1, 1, false, "linear"),
      convolution_channels(convolution_channels),
      memory_channels(Activations::isGated(activation) ? convolution_channels * 2 : convolution_channels),
      activation(activation),
      dilations(dilations)
{
}

void WaveNetAR::prepare(int buffer_size)
{
    conv_data.resize(samples_per_block * memory_channels);
    skip_data.resize(samples_per_block * skip_channels);
    inputBuffer.resize(input_channels); // only single timestep
    outputBuffer.resize(output_channels); // only single timestep
    conv_stack.prepare(samples_per_block);
    this->reset();
}

void WaveNetAR::reset()
{
    conv_stack.reset();
    input_layer.reset();
    output_layer1.reset();
    output_layer2.reset();
}

void WaveNetAR::process(float * const outputData, int total_samples)
{
    std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
    for (int i = 0; i < total_samples; i++)
    {
        // calculate offsets

        std::cerr << "timestep " << i << "/" << total_samples << std::endl; 
        // NOTE data must be channels major

        // always process just one sample
        input_layer.process(inputBuffer.data(), conv_data.data(), 1u);
        conv_stack.process(conv_data.data(), skip_data.data(), 1u);
        output_layer1.process(skip_data.data(), conv_data.data(), 1u);
        output_layer2.process(conv_data.data(), outputBuffer.data(), 1u);

        for (int t = 0; t < total_samples; t++)
        {
            std::cerr << outputData[t] << ", ";
        }
        std::cerr << std::endl;

        // copy output to input
        inputBuffer[0] = outputBuffer[0];
        outputData[i] = outputBuffer[0];

    }
}

void WaveNetAR::processConditional(const float *conditioning,
                                   float *const outputData, int total_samples)
{
    if (total_samples > samples_per_block)
        prepare(total_samples);
    input_layer.process(inputBuffer.data(), conv_data.data(), total_samples);
    conv_stack.processConditional(conv_data.data(), conditioning, skip_data.data(), total_samples);
    output_layer1.process(skip_data.data(), conv_data.data(), total_samples);
    output_layer2.process(conv_data.data(), outputData, total_samples);
}

inline int WaveNetAR::idx(int ch, int i, int total_samples)
{
    return ch * total_samples + i;
}

void WaveNetAR::setStackConvolutionWeight(const torch::Tensor &W, int layerIdx)
{
    conv_stack.setConvolutionWeight(W, layerIdx);
}

void WaveNetAR::setStackConvolutionBias(const torch::Tensor &b, int layerIdx)
{
    conv_stack.setConvolutionBias(b, layerIdx);
}

void WaveNetAR::setStackOutputWeight(const torch::Tensor &W, int layerIdx)
{
    conv_stack.setOutputWeight(W, layerIdx);
}

void WaveNetAR::setStackOutputBias(const torch::Tensor &b, int layerIdx)
{
    conv_stack.setOutputBias(b, layerIdx);
}

void WaveNetAR::setInputWeight(const torch::Tensor &W)
{
    input_layer.setConvolutionWeight(W);
}

void WaveNetAR::setInputBias(const torch::Tensor &b)
{
    input_layer.setConvolutionBias(b);
}

void WaveNetAR::setOutputWeight(const torch::Tensor &W, int layerIdx)
{
    if (layerIdx == 0)
        output_layer1.setConvolutionWeight(W);
    else if (layerIdx == 1)
        output_layer2.setConvolutionWeight(W);
}

void WaveNetAR::setOutputBias(const torch::Tensor &b, int layerIdx)
{
    if (layerIdx == 0)
        output_layer1.setConvolutionBias(b);
    else if (layerIdx == 1)
        output_layer2.setConvolutionBias(b);
}
