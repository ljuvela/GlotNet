#include "ConvolutionStack.h"

namespace glotnet
{


ConvolutionStack::ConvolutionStack(
    size_t num_channels, size_t num_skip_channels, size_t num_cond_channels,
    size_t filter_width, std::vector<int> dilations, std::string activation, bool residual)
    : dilations(dilations),
      num_layers(dilations.size()),
      residual(residual),
      num_channels(num_channels),
      num_skip_channels(num_skip_channels),
      num_cond_channels(num_cond_channels),
      filter_width(filter_width),
      activation(activation)
{
    initLayers();
}

void ConvolutionStack::prepare(int buffer_size)
{
    samples_per_block = buffer_size;
    residual_data.resize(samples_per_block * num_channels);
}

void ConvolutionStack::reset()
{
    for (auto & l : layers)
        l.reset();
    std::fill(residual_data.begin(), residual_data.end(), 0.0f);
}

void ConvolutionStack::saveResidual(const float *data, int timesteps)
{
    for (size_t i = 0; i < timesteps * num_channels; ++i)
        residual_data[i] = data[i];
}

void ConvolutionStack::addResidual(float *data, int timesteps)
{
    for (size_t i = 0; i < timesteps * num_channels; ++i)
        data[i] += residual_data[i];
}

void ConvolutionStack::process(float *data, float* skip_data, int timesteps)
{
    const size_t num_layers = dilations.size();
    if (timesteps > samples_per_block)
        prepare(timesteps);
    for (int layer_idx = 0; layer_idx < num_layers; ++layer_idx)
    {
        if (residual)
            saveResidual(data, timesteps);
        float *skip_ptr = &skip_data[layer_idx * timesteps * num_skip_channels];
        layers[layer_idx].process(data, data, skip_ptr, timesteps);
        if (residual)
            addResidual(data, timesteps);
    }
}

// TODO merge functions, pass null conditioning by default
void ConvolutionStack::processConditional(float *data, const float *conditioning, float* skip_data, int timesteps)
{
    const size_t num_layers = dilations.size();
    if (timesteps > samples_per_block)
        prepare(timesteps);
    for (int layer_idx = 0; layer_idx < num_layers; layer_idx++)
    {
        if (residual)
            saveResidual(data, timesteps);
        float *skip_ptr = &skip_data[layer_idx * timesteps * num_skip_channels];
        layers[layer_idx].processConditional(data, conditioning,
                                             data, skip_ptr, timesteps);
        if (residual)
            addResidual(data, timesteps);
    }
}

void ConvolutionStack::setConvolutionWeight(const torch::Tensor &W, size_t layerIdx)
{
    layers[layerIdx].setConvolutionWeight(W);
}

void ConvolutionStack::setConvolutionBias(const torch::Tensor &b, size_t layerIdx)
{
    layers[layerIdx].setConvolutionBias(b);
}

void ConvolutionStack::setOutputWeight(const torch::Tensor &W, size_t layerIdx)
{
    layers[layerIdx].setOutputWeight(W);
}

void ConvolutionStack::setOutputBias(const torch::Tensor &b, size_t layerIdx)
{
    layers[layerIdx].setOutputBias(b);
}

void ConvolutionStack::setSkipWeight(const torch::Tensor &W, size_t layerIdx)
{
    layers[layerIdx].setSkipWeight(W);
}

void ConvolutionStack::setSkipBias(const torch::Tensor &b, size_t layerIdx)
{
    layers[layerIdx].setSkipBias(b);
}

void ConvolutionStack::setCondWeight(const torch::Tensor &W, size_t layerIdx)
{
    layers[layerIdx].setCondWeight(W);
}

void ConvolutionStack::setCondBias(const torch::Tensor &b, size_t layerIdx)
{
    layers[layerIdx].setCondBias(b);
}

void ConvolutionStack::initLayers()
{
    layers.clear();
    layers.reserve(dilations.size());
    for (size_t i = 0; i < dilations.size(); ++i)
    {
        bool use_output_transform = true;
        if (i == dilations.size() - 1)
            use_output_transform = false;
        layers.push_back(ConvolutionLayer(
            num_channels, num_channels,
            num_skip_channels, num_cond_channels,
            filter_width, dilations[i],
            use_output_transform, activation));
    }
}

} // namespace glotnet