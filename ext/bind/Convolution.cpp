#include <torch/extension.h>
#include <vector>

#include "../src/Convolution.h"

namespace glotnet
{
namespace bindings
{

class Convolution
{
public:
    Convolution(int64_t input_channels, int64_t output_channels, int64_t filter_width, int64_t dilation)
        : conv(input_channels, output_channels, filter_width, dilation),
        input_channels(input_channels),
        output_channels(output_channels),
        filter_width(filter_width),
        dilation(dilation)
    {
    }

    void setKernel(const torch::Tensor &weight)
    {
        assert (filter_width == weight.size(2));
        assert (output_channels == weight.size(0));
        conv.setKernel(weight);
    }

    void setBias(const torch::Tensor &bias)
    {
        assert (output_channels == bias.size(0));
        conv.setBias(bias);
    }

    std::vector<at::Tensor> forward(const torch::Tensor &input)
    {
        int64_t batch_size = input.size(0);
        int64_t timesteps = input.size(1);
        assert (input_channels == input.size(2));

        auto output = torch::zeros({batch_size, timesteps, output_channels});

        auto input_a  = input.accessor<float, 3>();  // size (batch, time, input_channels)
        auto output_a = output.accessor<float, 3>(); // size (batch, time, output_channels)

        for (int64_t b = 0; b < batch_size; b++)
        {
            conv.resetBuffer(); // TODO: multiprocessing with copies, no need to reset
            conv.process(&(input_a[b][0][0]),
                        &(output_a[b][0][0]),
                        timesteps);
        }

        return {output};
    }

private:

    glotnet::Convolution conv;
    int64_t input_channels;
    int64_t output_channels;
    int64_t filter_width;
    int64_t dilation;
};

} // namespace torch
} // namespace glotnet

namespace glotnet
{
namespace convolution
{

std::vector<at::Tensor> forward(
    const torch::Tensor &input,
    const torch::Tensor &weight,
    const torch::Tensor &bias,
    size_t dilation=1)
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(1);
    int64_t input_channels = input.size(2);

    int64_t filter_width = weight.size(2);
    int64_t output_channels = weight.size(0);
    int64_t bias_size = bias.size(0);

    assert (input_channels == weight.size(1));
    assert (bias_size == output_channels);

    auto output = torch::zeros({batch_size, timesteps, output_channels});

    auto input_a  = input.accessor<float, 3>();  // size (batch, time, input_channels)
    auto output_a = output.accessor<float, 3>(); // size (batch, time, output_channels)

    auto conv = Convolution(input_channels, output_channels, filter_width, dilation);
    conv.setKernel(weight);
    conv.setBias(bias);

    for (int64_t b = 0; b < batch_size; b++)
    {
        conv.resetBuffer();
        conv.process(&(input_a[b][0][0]),
                     &(output_a[b][0][0]),
                     timesteps);
    }

    return {output};
}


std::vector<at::Tensor> forward_cond(
    const torch::Tensor & input,
    const torch::Tensor & cond_input,
    const torch::Tensor & weight,
    const torch::Tensor & bias,
    size_t dilation=1)
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(1);
    int64_t input_channels = input.size(2);

    int64_t filter_width = weight.size(2);
    int64_t output_channels = weight.size(0);
    int64_t bias_size = bias.size(0);

    assert (input_channels == weight.size(1));
    assert (bias_size == output_channels);

    auto output = torch::zeros({batch_size, timesteps, output_channels});

    // TODO: use accessors instead of raw data pointers
    float * data_in = input.data_ptr<float>();
    float * data_cond = cond_input.data_ptr<float>();
    float * data_out = output.data_ptr<float>();

    auto conv = Convolution(input_channels, output_channels, filter_width, dilation);
    conv.setKernel(weight);
    conv.setBias(bias);

    for (int64_t b = 0; b < batch_size; b++)
    {
        conv.resetBuffer();
        conv.processConditional(&(data_in[b * input_channels * timesteps]),
                     &(data_cond[b * output_channels * timesteps]),
                     &(data_out[b * output_channels * timesteps]),
                     timesteps); // time first (rightmost)
    }

    return {output};
}

std::vector<at::Tensor> forward_autoregressive(
    const torch::Tensor & input,
    const torch::Tensor & weight,
    const torch::Tensor & bias,
    size_t dilation=1)
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(1);
    int64_t input_channels = input.size(2);

    int64_t filter_width = weight.size(2);
    int64_t output_channels = weight.size(0);
    int64_t bias_size = bias.size(0);

    assert (input_channels == weight.size(1));
    assert (bias_size == output_channels);

    auto output = torch::zeros({batch_size, timesteps, output_channels});

    auto input_a  = input.accessor<float, 3>();  // size (batch, time, input_channels)
    auto output_a = output.accessor<float, 3>(); // size (batch, time, output_channels)

    auto conv = ConvolutionAR(input_channels, output_channels, filter_width, dilation);
    conv.setKernel(weight);
    conv.setBias(bias);

    for (int64_t b = 0; b < batch_size; b++)
    {
        conv.resetBuffer();
        conv.process(&(input_a[b][0][0]),
                     &(output_a[b][0][0]),
                     timesteps);
    }

    return {output};
}


} // convolution
} // glotnet

void init_convolution(py::module &m)
{
    m.def("convolution_forward", &(glotnet::convolution::forward), "Convolution forward");
    m.def("convolution_cond_forward", &(glotnet::convolution::forward_cond), "Convolution conditional forward");
    m.def("convolution_forward_ar", &(glotnet::convolution::forward_autoregressive), "Convolution autoregressive forward");
    py::class_<glotnet::bindings::Convolution>(m, "Convolution")
        .def(py::init<int64_t, int64_t, int64_t, int64_t>())
        .def("set_kernel", &glotnet::bindings::Convolution::setKernel)
        .def("set_bias", &glotnet::bindings::Convolution::setBias)
        .def("forward", &glotnet::bindings::Convolution::forward);
}