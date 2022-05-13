#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/ConvolutionLayer.h"

namespace glotnet
{
namespace convolution_layer
{

std::vector<at::Tensor> forward(
    torch::Tensor &input,
    torch::Tensor &weight_conv,
    torch::Tensor &bias_conv,
    torch::Tensor &weight_out,
    torch::Tensor &bias_out,
    int dilation=1,
    bool use_output_transform=true,
    std::string activation_name="gated"
    )
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(1);

    int64_t filter_width = weight_conv.size(2);
    int64_t input_channels = weight_conv.size(1);

    int64_t output_channels = weight_out.size(1);

    // input channels
    assert(input.size(2) == weight_conv.size(1));

    const int skip_channels = 0;
    const int cond_channels = 0;
    auto layer = ConvolutionLayer(input_channels, output_channels, skip_channels, cond_channels,
                                  filter_width, dilation, use_output_transform, activation_name);
    layer.setConvolutionWeight(weight_conv);
    layer.setConvolutionBias(bias_conv);
    if (use_output_transform)
    {
        layer.setOutputWeight(weight_out);
        layer.setOutputBias(bias_out);
    }

    auto output = torch::zeros({batch_size, timesteps, output_channels});
    float * data_in = input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();
    for (long long b = 0; b < batch_size; b++)
    {
        layer.reset();
        layer.process(&(data_in[b * input_channels * timesteps]),
                      &(data_out[b * output_channels * timesteps]),
                      timesteps);
    }
    return {output};
}

std::vector<at::Tensor> skip_forward(
    torch::Tensor &input,
    torch::Tensor &weight_conv,
    torch::Tensor &bias_conv,
    torch::Tensor &weight_out,
    torch::Tensor &bias_out,
    torch::Tensor &weight_skip,
    torch::Tensor &bias_skip,
    int dilation=1,
    bool use_output_transform=true,
    std::string activation_name="gated"
    )
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(1);

    int64_t filter_width = weight_conv.size(2);
    int64_t input_channels = weight_conv.size(1);

    int64_t output_channels = weight_out.size(0);
    int64_t skip_channels = weight_skip.size(0);
    int64_t cond_channels = 0;

    // input channels
    assert(input.size(2) == weight_conv.size(1));

    auto layer = ConvolutionLayer(input_channels, output_channels, skip_channels, cond_channels,
                                  filter_width, dilation, use_output_transform, activation_name);
    layer.setConvolutionWeight(weight_conv);
    layer.setConvolutionBias(bias_conv);
    if (use_output_transform)
    {
        layer.setOutputWeight(weight_out);
        layer.setOutputBias(bias_out);
    }
    layer.setSkipWeight(weight_skip);
    layer.setSkipBias(bias_skip);

    auto output = torch::zeros({batch_size, timesteps, output_channels});
    auto skip = torch::zeros({batch_size, timesteps, skip_channels});
    float * data_in = input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();
    float * data_skip = skip.data_ptr<float>();
    for (long long b = 0; b < batch_size; b++)
    {
        layer.reset();
        layer.process(&(data_in[b * input_channels * timesteps]),
                      &(data_out[b * output_channels * timesteps]),
                      &(data_skip[b * skip_channels * timesteps]),
                      timesteps); // time first (rightmost)
    }
    return {output, skip};
}

std::vector<at::Tensor> skip_cond_forward(
    torch::Tensor &input,
    torch::Tensor &cond_input,
    torch::Tensor &weight_conv,
    torch::Tensor &bias_conv,
    torch::Tensor &weight_out,
    torch::Tensor &bias_out,
    torch::Tensor &weight_skip,
    torch::Tensor &bias_skip,
    torch::Tensor &weight_cond,
    torch::Tensor &bias_cond,
    int dilation=1,
    bool use_output_transform=true,
    std::string activation_name="gated"
    )
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(1);

    int64_t filter_width = weight_conv.size(2);
    int64_t input_channels = weight_conv.size(1);
    int64_t output_channels = weight_out.size(0);
    int64_t skip_channels = weight_skip.size(0);
    int64_t cond_channels = cond_input.size(2);

    // input channels
    assert(input.size(2) == weight_conv.size(1));
    // cond channels
    assert(cond_input.size(2) == weight_cond.size(1));

    auto layer = ConvolutionLayer(input_channels, output_channels,
                                  skip_channels, cond_channels,
                                  filter_width, dilation,
                                  use_output_transform, activation_name);
    // convolution
    layer.setConvolutionWeight(weight_conv);
    layer.setConvolutionBias(bias_conv);
    // outputs
    if (use_output_transform)
    {
        layer.setOutputWeight(weight_out);
        layer.setOutputBias(bias_out);
    }
    // skips
    layer.setSkipWeight(weight_skip);
    layer.setSkipBias(bias_skip);
    // conditioning
    layer.setCondWeight(weight_cond);
    layer.setCondBias(bias_cond);

    auto output = torch::zeros({batch_size, timesteps, output_channels});
    auto skip = torch::zeros({batch_size, timesteps, skip_channels});
    float * data_in = input.data_ptr<float>();
    float * data_cond = cond_input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();
    float * data_skip = skip.data_ptr<float>();
    for (long long b = 0; b < batch_size; b++)
    {
        layer.reset();
        layer.processConditional(&(data_in[b * input_channels * timesteps]),
                                 &(data_cond[b * cond_channels * timesteps]),
                                 &(data_out[b * output_channels * timesteps]),
                                 &(data_skip[b * skip_channels * timesteps]),
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
    m.def("convolution_layer_skip_forward", &(glotnet::convolution_layer::skip_forward), "ConvolutionLayer skip forward");
    m.def("convolution_layer_skip_cond_forward", &(glotnet::convolution_layer::skip_cond_forward), "ConvolutionLayer conditional forward");
}