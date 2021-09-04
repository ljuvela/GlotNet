#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/Convolution.h"

namespace glotnet
{
namespace convolution
{

std::vector<at::Tensor> forward(
    torch::Tensor input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool training=true,
    int dilation=1)
{
    int64_t batch_size = input.size(0);
    int64_t input_channels = input.size(1);
    int64_t timesteps = input.size(2);

    int64_t filter_width = weight.size(0);
    int64_t output_channels = weight.size(2);

   
    int64_t bias_size = bias.size(0);

    assert (input_channels == weight.size(1));
    assert (bias_size == output_channels);

    // auto output = 1.0 * input; // convenient copy
    auto output = torch::zeros({batch_size, output_channels, timesteps});

    float * data_in = input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();

    auto conv = Convolution(input_channels, output_channels, filter_width, dilation);
    conv.setKernel(weight.data_ptr<float>(), filter_width * input_channels * output_channels);
    conv.setBias(bias.data_ptr<float>(), bias_size);

    for (int64_t b = 0; b < batch_size; b++)
    {
        conv.resetFifo();
        conv.process(&(data_in[b * input_channels * timesteps]),
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

} // convolution
} // glotnet

void init_convolution(py::module &m)
{
    m.def("convolution_forward", &(glotnet::convolution::forward), "Convolution forward");
    m.def("convolution_backward", &(glotnet::convolution::backward), "Convolution backward");
}