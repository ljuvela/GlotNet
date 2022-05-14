#include "WaveNet.h"

WaveNet::WaveNet(size_t input_channels, size_t output_channels,
                 size_t convolution_channels, size_t skip_channels, size_t cond_channels,
                 size_t filter_width, std::string activation, std::vector<int> dilations)
    : input_channels(input_channels),
      output_channels(output_channels),
      cond_channels(cond_channels),
      filter_width(filter_width),
      num_layers(dilations.size()),
      skip_channels(skip_channels),
      conv_stack(convolution_channels, skip_channels, cond_channels, filter_width, dilations, activation),
      input_layer(input_channels, convolution_channels, 0, 0, 1, 1, false, "tanh"),
      output_layer1(skip_channels, convolution_channels, 0, 0, 1, 1, false, "tanh"),
      output_layer2(convolution_channels, output_channels, 0, 0, 1, 1, false, "linear"),
      convolution_channels(convolution_channels),
      activation(activation),
      memory_channels(Activations::isGated(activation) ? convolution_channels * 2 : convolution_channels),
      dilations(dilations)
{
}

void WaveNet::prepare(int timesteps_new)
{
    timesteps = timesteps_new;
    conv_data.resize(timesteps * memory_channels);
    skip_data.resize(num_layers * timesteps * skip_channels);
    skip_sum.resize(timesteps * skip_channels);
    input_layer.prepare(timesteps);
    conv_stack.prepare(timesteps);
    output_layer1.prepare(timesteps);
    output_layer2.prepare(timesteps);
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
    input_layer.process(inputData, conv_data.data(), timesteps);
    conv_stack.process(conv_data.data(), skip_data.data(), timesteps);
    reduceSkipSum(skip_data.data(), skip_sum.data());
    output_layer1.process(skip_sum.data(), conv_data.data(), timesteps);
    output_layer2.process(conv_data.data(), outputData, timesteps);
}

void WaveNet::processConditional(const float *inputData, const float *conditioning,
                                 float *outputData, int total_samples)
{
    input_layer.process(inputData, conv_data.data(), total_samples);
    conv_stack.processConditional(conv_data.data(), conditioning, skip_data.data(), total_samples);
    reduceSkipSum(skip_data.data(), skip_sum.data());
    output_layer1.process(skip_sum.data(), conv_data.data(), total_samples);
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

void WaveNet::setStackSkipWeight(const torch::Tensor &W, int layerIdx)
{
    conv_stack.setSkipWeight(W, layerIdx);
}

void WaveNet::setStackSkipBias(const torch::Tensor &b, int layerIdx)
{
    conv_stack.setSkipBias(b, layerIdx);
}

void WaveNet::setStackCondWeight(const torch::Tensor &W, int layerIdx)
{
    conv_stack.setCondWeight(W, layerIdx);
}

void WaveNet::setStackCondBias(const torch::Tensor &b, int layerIdx)
{
    conv_stack.setCondBias(b, layerIdx);
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

void WaveNet::reduceSkipSum(const float * skip_data, float * skip_sum)
{
    const size_t L = num_layers;
    const size_t T = timesteps;
    const size_t C = skip_channels;
    memset(skip_sum, 0.0f, sizeof(float) * T * C);
    for (size_t l = 0; l < L; l++)
        for (size_t t = 0; t < T; t++)
            for (size_t c = 0; c < C; c++)
                skip_sum[t * C + c] += skip_data[l * (T * C) + (t * C) + c];
}