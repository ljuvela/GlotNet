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
    Convolution(int64_t input_channels, int64_t output_channels,
                int64_t filter_width, int64_t dilation, bool use_film)
        : conv(input_channels, output_channels, filter_width, dilation),
          input_channels(input_channels),
          output_channels(output_channels),
          filter_width(filter_width),
          dilation(dilation),
          use_film(use_film)
    {
    }

    void setWeight(const torch::Tensor &weight)
    {
        assert (filter_width == weight.size(2));
        assert (output_channels == weight.size(0));
        conv.setWeight(weight);
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
        assert(input_channels == input.size(2));

        auto output = torch::zeros({batch_size, timesteps, output_channels});

        auto input_a = input.accessor<float, 3>();   // size (batch, time, input_channels)
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

    std::vector<at::Tensor> forward_cond(
        const torch::Tensor &input,
        const torch::Tensor &cond_input)
    {
        int64_t batch_size = input.size(0);
        int64_t timesteps = input.size(1);
        int64_t input_channels = input.size(2);

        assert(input_channels == weight.size(1));
        assert(bias_size == output_channels);

        auto output = torch::zeros({batch_size, timesteps, output_channels});

        auto input_a = input.accessor<float, 3>();     // size (batch, time, input_channels)
        auto cond_a = cond_input.accessor<float, 3>(); // size (batch, time, output_channels)
        auto output_a = output.accessor<float, 3>();   // size (batch, time, output_channels)

        for (int64_t b = 0; b < batch_size; b++)
        {
            conv.resetBuffer();
            conv.processConditional(&(input_a[b][0][0]),
                                    &(cond_a[b][0][0]),
                                    &(output_a[b][0][0]),
                                    timesteps, use_film);
        }

        return {output};
    }

private:

    glotnet::Convolution conv;
    int64_t input_channels;
    int64_t output_channels;
    int64_t filter_width;
    int64_t dilation;
    const bool use_film;
};

} // namespace torch
} // namespace glotnet

namespace glotnet
{
namespace convolution
{

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
    conv.setWeight(weight);
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
    m.def("convolution_forward_ar", &(glotnet::convolution::forward_autoregressive), "Convolution autoregressive forward");
    py::class_<glotnet::bindings::Convolution>(m, "Convolution")
        .def(py::init<int64_t, int64_t, int64_t, int64_t, bool>())
        .def("set_weight", &glotnet::bindings::Convolution::setWeight)
        .def("set_bias", &glotnet::bindings::Convolution::setBias)
        .def("forward", &glotnet::bindings::Convolution::forward)
        .def("forward_cond", &glotnet::bindings::Convolution::forward_cond);
}