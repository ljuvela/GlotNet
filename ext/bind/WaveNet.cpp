#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "TensorView.h"
#include "../src/WaveNet.h"

namespace glotnet
{
namespace wavenet
{

std::vector<at::Tensor> forward(
    torch::Tensor &input,
    std::vector<torch::Tensor> &stack_weights_conv,
    std::vector<torch::Tensor> &stack_biases_conv,
    std::vector<torch::Tensor> &stack_weights_out,
    std::vector<torch::Tensor> &stack_biases_out,
    std::vector<torch::Tensor> &stack_weights_skip,
    std::vector<torch::Tensor> &stack_biases_skip,
    torch::Tensor &input_weight,
    torch::Tensor &input_bias,
    std::vector<torch::Tensor> &output_weights,
    std::vector<torch::Tensor> &output_biases,
    std::vector<int> &dilations,
    bool use_residual=true,
    std::string activation="gated"
    )
{
    const int64_t batch_size = input.size(0);
    const int64_t timesteps = input.size(1);
    const int64_t channels = input.size(2);

    const int64_t filter_width = stack_weights_conv[0].size(2);
    const int64_t residual_channels = stack_weights_conv[0].size(1);
    const int64_t skip_channels = stack_weights_skip[0].size(0);
    const int64_t input_channels = input_weight.size(1);
    const int64_t output_channels = output_weights.back().size(0); 
    const int64_t cond_channels = 0;

    // Instantiate model
    auto wavenet = WaveNet(input_channels, output_channels,
                           residual_channels, skip_channels, cond_channels,
                           filter_width, activation, dilations);

    // Set buffer size to match timesteps
    wavenet.prepare(timesteps);

    // Set parameters
    wavenet.setInputWeight(glotnet::view(input_weight));
    wavenet.setInputBias(glotnet::view(input_bias));
    int num_layers = dilations.size();
    for (size_t i = 0; i < num_layers; i++)
    {
        wavenet.setStackConvolutionWeight(glotnet::view(stack_weights_conv[i]), i);
        wavenet.setStackConvolutionBias(glotnet::view(stack_biases_conv[i]), i);
        wavenet.setStackOutputWeight(glotnet::view(stack_weights_out[i]), i);
        wavenet.setStackOutputBias(glotnet::view(stack_biases_out[i]), i);
        wavenet.setStackSkipWeight(glotnet::view(stack_weights_skip[i]), i);
        wavenet.setStackSkipBias(glotnet::view(stack_biases_skip[i]), i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        wavenet.setOutputWeight(glotnet::view(output_weights[i]), i);
        wavenet.setOutputBias(glotnet::view(output_biases[i]), i);
    }

    auto output = torch::zeros({batch_size, timesteps, output_channels});
    float * data_in = input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        wavenet.reset();
        wavenet.process(&(data_in[b * timesteps * input_channels]),
                        &(data_out[b * timesteps * output_channels]),
                        timesteps);
    }
    return {output};
}

std::vector<at::Tensor> cond_forward(
    const torch::Tensor &input,
    const torch::Tensor &cond_input,
    const std::vector<torch::Tensor> &stack_weights_conv,
    const std::vector<torch::Tensor> &stack_biases_conv,
    const std::vector<torch::Tensor> &stack_weights_out,
    const std::vector<torch::Tensor> &stack_biases_out,
    const std::vector<torch::Tensor> &stack_weights_skip,
    const std::vector<torch::Tensor> &stack_biases_skip,
    const std::vector<torch::Tensor> &stack_weights_cond,
    const std::vector<torch::Tensor> &stack_biases_cond,
    const torch::Tensor &input_weight,
    const torch::Tensor &input_bias,
    const std::vector<torch::Tensor> &output_weights,
    const std::vector<torch::Tensor> &output_biases,
    const std::vector<int> &dilations,
    bool use_residual=true,
    const std::string activation="gated"
    )
{
    const int64_t batch_size = input.size(0);
    const int64_t timesteps = input.size(1);
    const int64_t channels = input.size(2);

    const int64_t filter_width = stack_weights_conv[0].size(2);
    const int64_t residual_channels = stack_weights_conv[0].size(1);
    const int64_t skip_channels = stack_weights_skip[0].size(0);
    const int64_t input_channels = input_weight.size(1);
    const int64_t output_channels = output_weights.back().size(0);
    const int64_t cond_channels = cond_input.size(2);

    // instantiate model
    auto wavenet = WaveNet(input_channels, output_channels,
                           residual_channels, skip_channels, cond_channels,
                           filter_width, activation, dilations);

    // Set buffer size to match timesteps
    wavenet.prepare(timesteps);

    // Set parameters
    wavenet.setInputWeight(glotnet::view(input_weight));
    wavenet.setInputBias(glotnet::view(input_bias));
    int num_layers = dilations.size();
    for (size_t i = 0; i < num_layers; i++)
    {
        wavenet.setStackConvolutionWeight(glotnet::view(stack_weights_conv[i]), i);
        wavenet.setStackConvolutionBias(glotnet::view(stack_biases_conv[i]), i);
        wavenet.setStackOutputWeight(glotnet::view(stack_weights_out[i]), i);
        wavenet.setStackOutputBias(glotnet::view(stack_biases_out[i]), i);
        wavenet.setStackSkipWeight(glotnet::view(stack_weights_skip[i]), i);
        wavenet.setStackSkipBias(glotnet::view(stack_biases_skip[i]), i);
        wavenet.setStackCondWeight(glotnet::view(stack_weights_cond[i]), i);
        wavenet.setStackCondBias(glotnet::view(stack_biases_cond[i]), i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        wavenet.setOutputWeight(glotnet::view(output_weights[i]), i);
        wavenet.setOutputBias(glotnet::view(output_biases[i]), i);
    }

    auto output = torch::zeros({batch_size, timesteps, output_channels});
    float * data_in = input.data_ptr<float>();
    float * data_cond = cond_input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        wavenet.reset();
        wavenet.processConditional(&(data_in[b * timesteps * input_channels]),
                                   &(data_cond[b * timesteps * cond_channels]),
                                   &(data_out[b * timesteps * output_channels]),
                                   timesteps);
    }
    return {output};
}

} // wavenet
} // glotnet

void init_wavenet(py::module &m)
{
    m.def("wavenet_forward", &(glotnet::wavenet::forward), "WaveNet forward");
    m.def("wavenet_cond_forward", &(glotnet::wavenet::cond_forward), "WaveNet conditional forward");
}