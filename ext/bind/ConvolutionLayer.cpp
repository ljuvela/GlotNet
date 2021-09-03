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
    bool use_output_transform=true,
    std::string activationName="gated"
    )
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(2);

    int64_t filter_width = weight_conv.size(0);
    int64_t input_channels = weight_conv.size(1);

    int64_t output_channels = weight_out.size(1);

    auto layer = ConvolutionLayer(input_channels, output_channels, filter_width, dilation, use_output_transform, activationName);
    layer.setConvolutionWeight(weight_conv.data_ptr<float>(), weight_conv.size(0) * weight_conv.size(1) * weight_conv.size(2));
    layer.setConvolutionBias(bias_conv.data_ptr<float>(), bias_conv.size(0));
    if (use_output_transform)
    {
        layer.setOutputWeight(weight_out.data_ptr<float>(), weight_out.size(0) * weight_out.size(1) * weight_out.size(2));
        layer.setOutputBias(bias_out.data_ptr<float>(), bias_out.size(0));
    }


    auto output = torch::zeros({batch_size, output_channels, timesteps});
    auto skip = torch::zeros({batch_size, output_channels, timesteps});
    float * data_in = input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();
    float * data_skip = skip.data_ptr<float>();
    for (long long b = 0; b < batch_size; b++)
    {
        layer.reset();
        layer.process(&(data_in[b * input_channels * timesteps]),
                      &(data_out[b * output_channels * timesteps]),
                      &(data_skip[b * output_channels * timesteps]),
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

} // convolution
} // glotnet

void init_convolution_layer(py::module &m)
{
    m.def("convolution_layer_forward", &(glotnet::convolution_layer::forward), "ConvolutionLayer forward");
    m.def("convolution_layer_backward", &(glotnet::convolution_layer::backward), "ConvolutionLayer backward");
}