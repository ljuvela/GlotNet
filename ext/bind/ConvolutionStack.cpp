#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/ConvolutionStack.h"

namespace glotnet
{
namespace convolution_stack
{

std::vector<at::Tensor> forward(
    torch::Tensor input,
    std::vector<torch::Tensor> weights_conv,
    std::vector<torch::Tensor> biases_conv,
    std::vector<torch::Tensor> weights_out,
    std::vector<torch::Tensor> biases_out,
    std::vector<int> dilations,
    bool training=false,
    bool use_residual=true,
    std::string activation="gated"
    )
{
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t timesteps = input.size(2);
    int64_t filter_width = weights_conv[0].size(2);

    int num_layers = dilations.size();

    auto stack = ConvolutionStack(channels, filter_width, dilations, activation, use_residual);

    for (size_t i = 0; i < weights_conv.size(); i++)
        stack.setConvolutionWeight(weights_conv[i], i);

    for (size_t i = 0; i < biases_conv.size(); i++)
        stack.setConvolutionBias(biases_conv[i], i);

    for (size_t i = 0; i < weights_out.size(); i++)
        stack.setOutputWeight(weights_out[i], i);
    
    for (size_t i = 0; i < biases_out.size(); i++)
        stack.setOutputBias(biases_out[i], i);

    auto output = 1.0 * input;
    auto skip = torch::zeros({batch_size, num_layers * channels, timesteps});
    float * data = output.data_ptr<float>();
    float * data_skip = skip.data_ptr<float>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        stack.reset();
        stack.process(&(data[b * channels * timesteps]),
                      &(data_skip[b * channels * num_layers * timesteps]),
                      timesteps); // time first (rightmost)
    }
    return {output, skip};
}

std::vector<at::Tensor> cond_forward(
    torch::Tensor input,
    torch::Tensor cond_input,
    std::vector<torch::Tensor> weights_conv,
    std::vector<torch::Tensor> biases_conv,
    std::vector<torch::Tensor> weights_out,
    std::vector<torch::Tensor> biases_out,
    std::vector<int> dilations,
    bool training=false,
    bool use_residual=true,
    std::string activation="gated"
    )
{
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t timesteps = input.size(2);
    int64_t filter_width = weights_conv[0].size(2);

    int num_layers = dilations.size();

    auto stack = ConvolutionStack(channels, filter_width, dilations, activation, use_residual);

    for (size_t i = 0; i < weights_conv.size(); i++)
        stack.setConvolutionWeight(weights_conv[i], i);

    for (size_t i = 0; i < biases_conv.size(); i++)
        stack.setConvolutionBias(biases_conv[i], i);

    for (size_t i = 0; i < weights_out.size(); i++)
        stack.setOutputWeight(weights_out[i], i);
    
    for (size_t i = 0; i < biases_out.size(); i++)
        stack.setOutputBias(biases_out[i], i);

    // cond_input = 0.0 * cond_input;

    auto output = 1.0 * input;
    auto skip = torch::zeros({batch_size, num_layers * channels, timesteps});
    float * data = output.data_ptr<float>();
    float * data_cond = cond_input.data_ptr<float>();
    float * data_skip = skip.data_ptr<float>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        stack.reset();
        stack.processConditional(&(data[b * channels * timesteps]),
                                 &(data_cond[b * channels * num_layers * timesteps]),
                                 &(data_skip[b * channels * num_layers * timesteps]),
                                 timesteps); // time first (rightmost)
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