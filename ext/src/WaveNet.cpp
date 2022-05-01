#include "WaveNet.h"

WaveNet::WaveNet(size_t input_channels, size_t output_channels,
                 size_t convolution_channels, size_t skip_channels, size_t cond_channels,
                 size_t filter_width, std::string activation, std::vector<int> dilations)
    : input_channels(input_channels),
      output_channels(output_channels),
      filter_width(filter_width),
      skip_channels(convolution_channels * (int)dilations.size()),
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

void WaveNet::prepare(int buffer_size)
{
    samples_per_block = buffer_size;
    conv_data.resize(samples_per_block * memory_channels);
    skip_data.resize(samples_per_block * skip_channels);
    conv_stack.prepare(samples_per_block);
    this->reset();
}

void WaveNet::reset()
{
    conv_stack.reset();
    input_layer.reset();
    output_layer1.reset();
    output_layer2.reset();
}

void WaveNet::process(const float *inputData, float *outputData, int total_samples)
{
    if (total_samples > samples_per_block)
        prepare(total_samples);
    input_layer.process(inputData, conv_data.data(), total_samples);
    conv_stack.process(conv_data.data(), skip_data.data(), total_samples);
    output_layer1.process(skip_data.data(), conv_data.data(), total_samples);
    output_layer2.process(conv_data.data(), outputData, total_samples);
}

void WaveNet::processConditional(const float *inputData, const float *conditioning,
                                 float *outputData, int total_samples)
{
    if (total_samples > samples_per_block)
        prepare(total_samples);
    input_layer.process(inputData, conv_data.data(), total_samples);
    conv_stack.processConditional(conv_data.data(), conditioning, skip_data.data(), total_samples);
    output_layer1.process(skip_data.data(), conv_data.data(), total_samples);
    output_layer2.process(conv_data.data(), outputData, total_samples);
}

inline int WaveNet::idx(int ch, int i, int total_samples)
{
    return ch * total_samples + i;
}

void WaveNet::setStackConvolutionWeight(const torch::Tensor &W, int layerIdx)
{
    conv_stack.setConvolutionWeight(W, layerIdx);
}

void WaveNet::setStackConvolutionBias(const torch::Tensor &b, int layerIdx)
{
    conv_stack.setConvolutionBias(b, layerIdx);
}

void WaveNet::setStackOutputWeight(const torch::Tensor &W, int layerIdx)
{
    conv_stack.setOutputWeight(W, layerIdx);
}

void WaveNet::setStackOutputBias(const torch::Tensor &b, int layerIdx)
{
    conv_stack.setOutputBias(b, layerIdx);
}

void WaveNet::setInputWeight(const torch::Tensor &W)
{
    input_layer.setConvolutionWeight(W);
}

void WaveNet::setInputBias(const torch::Tensor &b)
{
    input_layer.setConvolutionBias(b);
}

void WaveNet::setOutputWeight(const torch::Tensor &W, int layerIdx)
{
    if (layerIdx == 0)
        output_layer1.setConvolutionWeight(W);
    else if (layerIdx == 1)
        output_layer2.setConvolutionWeight(W);
}

void WaveNet::setOutputBias(const torch::Tensor &b, int layerIdx)
{
    if (layerIdx == 0)
        output_layer1.setConvolutionBias(b);
    else if (layerIdx == 1)
        output_layer2.setConvolutionBias(b);
}
