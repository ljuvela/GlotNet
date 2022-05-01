#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/ConvolutionStack.h"

namespace glotnet
{
namespace convolution_stack
{

std::vector<at::Tensor> forward(
    torch::Tensor &input,
    std::vector<torch::Tensor> &weights_conv,
    std::vector<torch::Tensor> &biases_conv,
    std::vector<torch::Tensor> &weights_out,
    std::vector<torch::Tensor> &biases_out,
    std::vector<torch::Tensor> &weights_skip,
    std::vector<torch::Tensor> &biases_skip,
    std::vector<int> &dilations,
    bool training=false,
    bool use_residual=true,
    std::string activation="gated"
    )
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(1);
    int64_t channels = input.size(2);
    int64_t skip_channels = weights_skip[0].size(0);
    int64_t filter_width = weights_conv[0].size(2);
    const int num_layers = dilations.size();
    const size_t cond_channels = 0;

    auto stack = ConvolutionStack(channels, skip_channels, cond_channels,
                                  filter_width, dilations, activation, use_residual);

    for (size_t i = 0; i < weights_conv.size(); i++)
        stack.setConvolutionWeight(weights_conv[i], i);

    for (size_t i = 0; i < biases_conv.size(); i++)
        stack.setConvolutionBias(biases_conv[i], i);

    for (size_t i = 0; i < weights_out.size(); i++)
        stack.setOutputWeight(weights_out[i], i);
    
    for (size_t i = 0; i < biases_out.size(); i++)
        stack.setOutputBias(biases_out[i], i);

    for (size_t i = 0; i < weights_skip.size(); i++)
        stack.setSkipWeight(weights_skip[i], i);
    
    for (size_t i = 0; i < biases_skip.size(); i++)
        stack.setSkipBias(biases_skip[i], i);

    auto output = 1.0 * input; // clone
    auto skip = torch::zeros({batch_size, num_layers, timesteps, skip_channels});
    float * data = output.data_ptr<float>();
    float * data_skip = skip.data_ptr<float>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        stack.reset();
        stack.process(&(data[b * timesteps * channels]),
                      &(data_skip[b * num_layers * timesteps * skip_channels]),
                      timesteps);
    }
    return {output, skip};
}

std::vector<at::Tensor> cond_forward(
    const torch::Tensor &input,
    const torch::Tensor &cond_input,
    const std::vector<torch::Tensor> &weights_conv,
    const std::vector<torch::Tensor> &biases_conv,
    const std::vector<torch::Tensor> &weights_out,
    const std::vector<torch::Tensor> &biases_out,
    const std::vector<torch::Tensor> &weights_skip,
    const std::vector<torch::Tensor> &biases_skip,
    const std::vector<torch::Tensor> &weights_cond,
    const std::vector<torch::Tensor> &biases_cond,
    const std::vector<int> &dilations,
    bool training=false,
    bool use_residual=true,
    const std::string activation="gated"
    )
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(1);
    int64_t channels = input.size(2);
    int64_t cond_channels = cond_input.size(2);
    int64_t skip_channels = weights_skip[0].size(0);
    int64_t filter_width = weights_conv[0].size(2);
    int num_layers = dilations.size();

    auto stack = ConvolutionStack(channels, skip_channels, cond_channels,
                                  filter_width, dilations, activation, use_residual);

    for (size_t i = 0; i < weights_conv.size(); i++)
        stack.setConvolutionWeight(weights_conv[i], i);

    for (size_t i = 0; i < biases_conv.size(); i++)
        stack.setConvolutionBias(biases_conv[i], i);

    for (size_t i = 0; i < weights_out.size(); i++)
        stack.setOutputWeight(weights_out[i], i);
    
    for (size_t i = 0; i < biases_out.size(); i++)
        stack.setOutputBias(biases_out[i], i);

    for (size_t i = 0; i < weights_skip.size(); i++)
        stack.setSkipWeight(weights_skip[i], i);
    
    for (size_t i = 0; i < biases_skip.size(); i++)
        stack.setSkipBias(biases_skip[i], i);

    for (size_t i = 0; i < weights_cond.size(); i++)
        stack.setCondWeight(weights_cond[i], i);
    
    for (size_t i = 0; i < biases_cond.size(); i++)
        stack.setCondBias(biases_cond[i], i);

    auto output = 1.0 * input;
    auto skip = torch::zeros({batch_size, num_layers, timesteps, skip_channels});
    float * data = output.data_ptr<float>();
    float * data_cond = cond_input.data_ptr<float>();
    float * data_skip = skip.data_ptr<float>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        stack.reset();
        stack.processConditional(&(data[b * timesteps * channels]),
                                 &(data_cond[b * timesteps * cond_channels]),
                                 &(data_skip[b * num_layers * timesteps * skip_channels]),
                                 timesteps);
    }
    return {output, skip};
}

std::vector<torch::Tensor> backward(
    torch::Tensor d_output,
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias)
{
    // Get sizes of input tensor
    long long batch_size = input.size(0);
    long long channels = input.size(1);
    long long timesteps = input.size(2);

    // Placeholder identity backward pass
    auto d_input = 0.0 * input;
    auto d_weight = 0.0 * weight;
    auto d_bias = 0.0 * bias;

    return {d_input, d_weight, d_bias};
}

} // convolution_stack
} // glotnet

void init_convolution_stack(py::module &m)
{
    m.def("convolution_stack_forward", &(glotnet::convolution_stack::forward), "ConvolutionStack forward");
    m.def("convolution_stack_cond_forward", &(glotnet::convolution_stack::cond_forward), "ConvolutionStack conditional forward");
    m.def("convolution_stack_backward", &(glotnet::convolution_stack::backward), "ConvolutionStack backward");
}