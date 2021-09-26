#include <torch/extension.h>
#include <vector>

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
    int64_t filter_width = weight.size(2);
    int64_t output_channels = weight.size(0);
    int64_t bias_size = bias.size(0);

    assert (input_channels == weight.size(1));
    assert (bias_size == output_channels);

    auto output = torch::zeros({batch_size, output_channels, timesteps});

    float * data_in = input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();

    auto conv = Convolution(input_channels, output_channels, filter_width, dilation);
    conv.setKernel(weight);
    conv.setBias(bias);

    for (int64_t b = 0; b < batch_size; b++)
    {
        conv.resetFifo();
        conv.process(&(data_in[b * input_channels * timesteps]),
                     &(data_out[b * output_channels * timesteps]),
                     timesteps); // time first (rightmost)
    }

    return {output};
}


std::vector<at::Tensor> forward_cond(
    torch::Tensor input,
    torch::Tensor cond_input,
    torch::Tensor weight,
    torch::Tensor bias,
    bool training=true,
    int dilation=1)
{
    int64_t batch_size = input.size(0);
    int64_t input_channels = input.size(1);
    int64_t timesteps = input.size(2);
    int64_t filter_width = weight.size(2);
    int64_t output_channels = weight.size(0);
    int64_t bias_size = bias.size(0);

    assert (input_channels == weight.size(1));
    assert (bias_size == output_channels);

    auto output = torch::zeros({batch_size, output_channels, timesteps});

    float * data_in = input.data_ptr<float>();
    float * data_cond = cond_input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();

    auto conv = Convolution(input_channels, output_channels, filter_width, dilation);
    conv.setKernel(weight);
    conv.setBias(bias);

    for (int64_t b = 0; b < batch_size; b++)
    {
        conv.resetFifo();
        conv.processConditional(&(data_in[b * input_channels * timesteps]),
                     &(data_cond[b * output_channels * timesteps]),
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
    m.def("convolution_cond_forward", &(glotnet::convolution::forward_cond), "Convolution conditional forward");
    m.def("convolution_backward", &(glotnet::convolution::backward), "Convolution backward");
    //m.def("convolution_cond_backward", &(glotnet::convolution::backward_cond), "Convolution conditional backward");
}