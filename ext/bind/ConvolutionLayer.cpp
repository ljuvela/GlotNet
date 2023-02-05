#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/ConvolutionLayer.h"

using TensorRefList = std::vector<std::reference_wrapper<torch::Tensor>>;

namespace glotnet
{
namespace bindings
{

class ConvolutionLayer
{
public:
    ConvolutionLayer(int64_t input_channels, int64_t output_channels,
                     int64_t skip_channels, int64_t cond_channels,
                     int64_t filter_width, int64_t dilation=1,
                     bool use_output_transform=true, bool use_film=false,
                     const std::string &activation_name="gated")
        : layer(input_channels, output_channels,
                skip_channels, cond_channels,
                filter_width, dilation, 
                use_output_transform,
                activation_name),
          input_channels(input_channels),
          output_channels(output_channels),
          skip_channels(skip_channels),
          cond_channels(cond_channels),
          filter_width(filter_width),
          dilation(dilation),
          use_output_transform(use_output_transform),
          use_film(use_film),
          activation_name(activation_name)
    {
    }

    void setConvolutionWeight(const torch::Tensor &weight)
    {
        assert(filter_width == weight.size(2));
        assert(input_channels == weight.size(1));
        layer.setConvolutionWeight(weight);
    }

    void setConvolutionBias(const torch::Tensor &bias)
    {
        assert(output_channels == bias.size(0));
        layer.setConvolutionBias(bias);
    }

    void setOutputWeight(const torch::Tensor &weight)
    {
        assert(output_channels == weight.size(1));
        layer.setOutputWeight(weight);
    }

    void setOutputBias(const torch::Tensor &bias)
    {
        assert(output_channels == bias.size(0));
        layer.setOutputBias(bias);
    }

    void setSkipWeight(const torch::Tensor &weight)
    {
        assert(skip_channels == weight.size(0));
        layer.setSkipWeight(weight);
    }

    void setSkipBias(const torch::Tensor &bias)
    {
        assert(output_channels == bias.size(0));
        layer.setSkipBias(bias);
    }

    void setCondWeight(const torch::Tensor &weight)
    {
        assert(cond_channels == weight.size(0));
        layer.setCondWeight(weight);
    }

    void setCondBias(const torch::Tensor &bias)
    {
        assert(output_channels == bias.size(0));
        layer.setCondBias(bias);
    }
    
    void setParams(TensorRefList &params)
    {
        setConvolutionWeight(params[0]);
        setConvolutionBias(params[1]);
        if (use_output_transform)
        {
            setOutputWeight(params[2]);
            setOutputBias(params[3]);
        }
        if (skip_channels > 0)
        {
            setSkipWeight(params[4]);
            setSkipBias(params[5]);
        }
        if (cond_channels > 0)
        {
            setCondWeight(params[6]);
            setCondBias(params[7]);
        }
    }


    std::vector<at::Tensor> forward(
        torch::Tensor &input)
    {
        int64_t batch_size = input.size(0);
        int64_t timesteps = input.size(1);

        auto output = torch::zeros({batch_size, timesteps, output_channels});
        auto input_a = input.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        for (long long b = 0; b < batch_size; b++)
        {
            layer.reset();
            layer.process(&input_a[b][0][0],
                          &output_a[b][0][0],
                          timesteps);
        }
        return {output};
    }

    std::vector<at::Tensor> skip_forward(
        torch::Tensor &input)
    {
        int64_t batch_size = input.size(0);
        int64_t timesteps = input.size(1);

        // input channels
        assert(input.size(2) == weight_conv.size(1));

        auto output = torch::zeros({batch_size, timesteps, output_channels});
        auto skip = torch::zeros({batch_size, timesteps, skip_channels});
        auto input_a = input.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        auto skip_a = skip.accessor<float, 3>();
        for (long long b = 0; b < batch_size; b++)
        {
            layer.reset();
            layer.process(&input_a[b][0][0],
                          &output_a[b][0][0],
                          &skip_a[b][0][0],
                          timesteps); // time first (rightmost)
        }
        return {output, skip};
    }

    std::vector<at::Tensor> skip_cond_forward(
        torch::Tensor &input,
        torch::Tensor &cond_input)
    {
        int64_t batch_size = input.size(0);
        int64_t timesteps = input.size(1);

        // input channels
        assert(input.size(2) == weight_conv.size(1));
        // cond channels
        assert(cond_input.size(2) == weight_cond.size(1));

        auto output = torch::zeros({batch_size, timesteps, output_channels});
        auto skip = torch::zeros({batch_size, timesteps, skip_channels});
        auto input_a = input.accessor<float, 3>();
        auto cond_a = cond_input.accessor<float, 3>();
        auto output_a = output.accessor<float, 3>();
        auto skip_a = skip.accessor<float, 3>();
        for (long long b = 0; b < batch_size; b++)
        {
            layer.reset();
            layer.processConditional(
                &input_a[b][0][0],
                &cond_a[b][0][0],
                &output_a[b][0][0],
                &skip_a[b][0][0],
                timesteps);
        }
        return {output, skip};
    }

private:
    glotnet::ConvolutionLayer layer;
    const int64_t input_channels;
    const int64_t output_channels;
    const int64_t filter_width;
    const int64_t dilation;
    const int64_t skip_channels;
    const int64_t cond_channels;
    const bool use_output_transform;
    const bool use_film;
    std::string activation_name;
};


} // bindings
} // glotnet




void init_convolution_layer(py::module &m)
{

    py::class_<glotnet::bindings::ConvolutionLayer>(m, "ConvolutionLayer")
        .def(py::init<int64_t, int64_t,
                    int64_t, int64_t,
                    int64_t, int64_t,
                    bool, bool,
                    const std::string &>())
        .def("set_convolution_weight", &glotnet::bindings::ConvolutionLayer::setConvolutionWeight)
        .def("set_convolution_bias", &glotnet::bindings::ConvolutionLayer::setConvolutionBias)
        .def("set_output_weight", &glotnet::bindings::ConvolutionLayer::setOutputWeight)
        .def("set_output_bias", &glotnet::bindings::ConvolutionLayer::setOutputBias)
        .def("set_skip_weight", &glotnet::bindings::ConvolutionLayer::setSkipWeight)
        .def("set_skip_bias", &glotnet::bindings::ConvolutionLayer::setSkipBias)
        .def("set_cond_weight", &glotnet::bindings::ConvolutionLayer::setCondWeight)
        .def("set_cond_bias", &glotnet::bindings::ConvolutionLayer::setCondBias)
        .def("set_params", &glotnet::bindings::ConvolutionLayer::setParams)
        .def("forward", &glotnet::bindings::ConvolutionLayer::forward)
        .def("skip_forward", &glotnet::bindings::ConvolutionLayer::skip_forward)
        .def("skip_cond_forward", &glotnet::bindings::ConvolutionLayer::skip_cond_forward);

}