#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/WaveNetAR.h"

namespace glotnet
{
namespace wavenet_ar
{

using glotnet::WaveNetAR;

std::vector<at::Tensor> forward(
    int64_t timesteps,
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
    std::string activation="gated",
    float temperature=1.0
    )
{
    const int64_t batch_size = 1;

    const int64_t filter_width = stack_weights_conv[0].size(2);
    const int64_t residual_channels = stack_weights_conv[0].size(1);
    const int64_t skip_channels = stack_weights_skip[0].size(0);
    const int64_t input_channels = input_weight.size(1);
    const int64_t output_channels = output_weights.back().size(0); 
    const int64_t cond_channels = 0;

    // Instantiate model
    auto wavenet = WaveNetAR(input_channels, output_channels,
                             residual_channels, skip_channels, cond_channels,
                             filter_width, activation, dilations);
    wavenet.prepare();
    wavenet.setDistribution("gaussian");
    wavenet.setSamplingTemperature(temperature);

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
        wavenet.setStackSkipWeight(stack_weights_skip[i], i);
        wavenet.setStackSkipBias(stack_biases_skip[i], i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        wavenet.setOutputWeight(output_weights[i], i);
        wavenet.setOutputBias(output_biases[i], i);
    }

    const auto audio_channels = input_channels;
    auto output = torch::zeros({batch_size, timesteps, audio_channels});
    auto output_a = output.accessor<float, 3>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        wavenet.reset();
        wavenet.process(&output_a[b][0][0], timesteps);
    }
    return {output};
}

std::vector<at::Tensor> cond_forward(
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
    const std::string activation="gated",
    float temperature=1.0
    )
{
    const int64_t batch_size = cond_input.size(0);
    const int64_t timesteps = cond_input.size(1);
    const int64_t cond_channels = cond_input.size(2);

    const int64_t filter_width = stack_weights_conv[0].size(2);
    const int64_t residual_channels = stack_weights_conv[0].size(1);
    const int64_t skip_channels = stack_weights_skip[0].size(0);
    const int64_t input_channels = input_weight.size(1);
    const int64_t output_channels = output_weights.back().size(0);

    // instantiate model
    auto wavenet = WaveNetAR(input_channels, output_channels,
                             residual_channels, skip_channels, cond_channels,
                             filter_width, activation, dilations);

    // Set buffer size to match timesteps
    wavenet.prepare();
    wavenet.setDistribution("gaussian");
    wavenet.setSamplingTemperature(temperature);

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
        wavenet.setStackSkipWeight(stack_weights_skip[i], i);
        wavenet.setStackSkipBias(stack_biases_skip[i], i);
        wavenet.setStackCondWeight(stack_weights_cond[i], i);
        wavenet.setStackCondBias(stack_biases_cond[i], i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        wavenet.setOutputWeight(output_weights[i], i);
        wavenet.setOutputBias(output_biases[i], i);
    }

    const auto audio_channels = input_channels;
    auto output =  torch::zeros({batch_size, timesteps, audio_channels});
    const auto cond_input_a = cond_input.accessor<float, 3>();
    auto output_a = output.accessor<float, 3>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        wavenet.reset();
        wavenet.processConditional(&cond_input_a[b][0][0],
                                   &output_a[b][0][0],
                                   timesteps);
    }
    return {output};
}

} // wavenet
} // glotnet

void init_wavenet_ar(py::module &m)
{
    m.def("wavenet_ar_forward", &(glotnet::wavenet_ar::forward), "WaveNet autoregressive forward");
    m.def("wavenet_ar_cond_forward", &(glotnet::wavenet_ar::cond_forward), "WaveNet autoregressive conditional forward");
}