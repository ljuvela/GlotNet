#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/ConvolutionLayer.h"

namespace glotnet
{
namespace convolution_layer
{

std::vector<at::Tensor> forward(
    torch::Tensor input,
    torch::Tensor weight_conv,
    torch::Tensor bias_conv,
    torch::Tensor weight_out,
    torch::Tensor bias_out,
    bool training=true,
    int dilation=1,
    bool use_residual=true,
    std::string activationName="gated"
    )
{
    int64_t batch_size = input.size(0);
    int64_t channels = input.size(1);
    int64_t timesteps = input.size(2);

    int64_t filter_width = weight_conv.size(0);
    int64_t output_channels = weight_out.size(1);
    int64_t input_channels = weight_conv.size(2);

    std::cerr << "Making Conv Layer" << std::endl;
    auto layer = ConvolutionLayer(input_channels, output_channels, filter_width, dilation, use_residual, activationName);
    std::cerr << "Set conv weights" << std::endl;
    // layer.setConvolutionWeight(weight_conv.data_ptr<float>(), weight_conv.size(0) * weight_conv.size(1) * weight_conv.size(2));
    std::cerr << "Set conv bias" << std::endl;
    // layer.setConvolutionBias(bias_conv.data_ptr<float>(), bias_conv.size(0));
    std::cerr << "Set out weights" << std::endl;
    // layer.setOutputWeight(weight_conv.data_ptr<float>(), weight_out.size(0) * weight_out.size(1) * weight_out.size(2));
    std::cerr << "Set out bias" << std::endl;
    // layer.setOutputBias(bias_out.data_ptr<float>(), bias_out.size(0));
    std::cerr << "All ok" << std::endl;

    auto output = 1.0 * input; // convenient copy
    auto skip = 0.0 * input;
    float * data_out = output.data_ptr<float>();
    float * data_skip = skip.data_ptr<float>();
    std::cerr << "Batch size " << batch_size << std::endl;
    for (long long b = 0; b < batch_size; b++)
    {
        std::cerr << "Processing " << std::endl;
        // TODO: flush layer memory
        // layer.process(&(data_out[b * channels * timesteps]),
        //               &(data_skip[b * channels * timesteps]),
        //               timesteps); // time first (rightmost)
        std::cerr << "Ok" << std::endl;
    }
    std::cerr << "Returning" << std::endl;
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

} // convolution
} // glotnet

void init_convolution_layer(py::module &m)
{
    m.def("convolution_layer_forward", &(glotnet::convolution_layer::forward), "ConvolutionLayer forward");
    m.def("convolution_layer_backward", &(glotnet::convolution_layer::backward), "ConvolutionLayer backward");
}