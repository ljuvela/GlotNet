#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/WaveNetAR.h"

namespace glotnet
{
namespace wavenet_ar
{

std::vector<at::Tensor> forward(
    int64_t timesteps,
    std::vector<torch::Tensor> stack_weights_conv,
    std::vector<torch::Tensor> stack_biases_conv,
    std::vector<torch::Tensor> stack_weights_out,
    std::vector<torch::Tensor> stack_biases_out,
    torch::Tensor input_weight,
    torch::Tensor input_bias,
    std::vector<torch::Tensor> output_weights,
    std::vector<torch::Tensor> output_biases,
    std::vector<int> dilations,
    bool training=false,
    bool use_residual=true,
    std::string activation="gated"
    )
{
    int64_t batch_size = 1;

    int64_t filter_width = stack_weights_conv[0].size(2);
    int64_t residual_channels = stack_weights_conv[0].size(1);
    int64_t input_channels = input_weight.size(1);
    int64_t output_channels = output_weights.back().size(0); 
    
    // instantiate model
    auto wavenet = WaveNetAR(input_channels, output_channels, residual_channels,
                           filter_width, activation, dilations);

    // Set buffer size to match timesteps
    wavenet.prepare(timesteps);

    // Set parameters
    wavenet.setInputWeight(input_weight);
    wavenet.setInputBias(input_bias);
    int num_layers = dilations.size();
    for (size_t i = 0; i < num_layers; i++)
    {
        wavenet.setStackConvolutionWeight(stack_weights_conv[i], i);
        wavenet.setStackConvolutionBias(stack_biases_conv[i], i);
        wavenet.setStackOutputWeight(stack_weights_out[i], i);
        wavenet.setStackOutputBias(stack_biases_out[i], i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        wavenet.setOutputWeight(output_weights[i], i);
        wavenet.setOutputBias(output_biases[i], i);
    }

    auto output =  torch::zeros({batch_size, output_channels, timesteps});
    float * data_out = output.data_ptr<float>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        // wavenet.reset();
        wavenet.process(&(data_out[b * output_channels * timesteps]),
                        timesteps); // time first (rightmost)
    }
    return {output};
}

std::vector<at::Tensor> cond_forward(
    torch::Tensor cond_input,
    std::vector<torch::Tensor> stack_weights_conv,
    std::vector<torch::Tensor> stack_biases_conv,
    std::vector<torch::Tensor> stack_weights_out,
    std::vector<torch::Tensor> stack_biases_out,
    torch::Tensor input_weight,
    torch::Tensor input_bias,
    std::vector<torch::Tensor> output_weights,
    std::vector<torch::Tensor> output_biases,
    std::vector<int> dilations,
    bool training=false,
    bool use_residual=true,
    std::string activation="gated"
    )
{
    int64_t batch_size = cond_input.size(0);
    int64_t channels = cond_input.size(1);
    int64_t timesteps = cond_input.size(2);

    int64_t filter_width = stack_weights_conv[0].size(2);
    int64_t residual_channels = stack_weights_conv[0].size(1);
    int64_t input_channels = input_weight.size(1);
    int64_t output_channels = output_weights.back().size(0); 
    
    // instantiate model
    auto wavenet = WaveNetAR(input_channels, output_channels, residual_channels,
                           filter_width, activation, dilations);

    // Set buffer size to match timesteps
    wavenet.prepare(timesteps);

    // Set parameters
    wavenet.setInputWeight(input_weight);
    wavenet.setInputBias(input_bias);
    int num_layers = dilations.size();
    for (size_t i = 0; i < num_layers; i++)
    {
        wavenet.setStackConvolutionWeight(stack_weights_conv[i], i);
        wavenet.setStackConvolutionBias(stack_biases_conv[i], i);
        wavenet.setStackOutputWeight(stack_weights_out[i], i);
        wavenet.setStackOutputBias(stack_biases_out[i], i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        wavenet.setOutputWeight(output_weights[i], i);
        wavenet.setOutputBias(output_biases[i], i);
    }

    auto output =  torch::zeros({batch_size, output_channels, timesteps});
    float * data_cond = cond_input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        wavenet.reset();
        wavenet.processConditional(&(data_cond[b * channels * num_layers * timesteps]),
                                   &(data_out[b * output_channels * timesteps]),
                                   timesteps); // time first (rightmost)
    }
    return {output};
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

} // wavenet
} // glotnet

void init_wavenet_ar(py::module &m)
{
    m.def("wavenet_ar_forward", &(glotnet::wavenet_ar::forward), "WaveNet autoregressive forward");
    m.def("wavenet_ar_cond_forward", &(glotnet::wavenet_ar::cond_forward), "WaveNet autoregressive conditional forward");
    m.def("wavenet_ar_backward", &(glotnet::wavenet_ar::backward), "WaveNet autoregressive backward");
}