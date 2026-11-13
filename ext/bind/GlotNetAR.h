#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "TensorView.h"
#include "glotnet/GlotNetAR.h"

namespace glotnet
{
namespace binding
{

class GlotNetAR
{

public:
    GlotNetAR(size_t input_channels, size_t output_channels,
              size_t convolution_channels, size_t skip_channels, size_t cond_channels,
              size_t filter_width, std::string activation, std::vector<int> dilations, 
              size_t lpc_order, bool sample_after_filtering)
        : input_channels(input_channels),
          output_channels(output_channels),
          convolution_channels(convolution_channels),
          skip_channels(skip_channels),
          cond_channels(cond_channels),
          filter_width(filter_width),
          activation(activation),
          dilations(dilations),
          lpc_order(lpc_order),
          model(input_channels, output_channels,
                convolution_channels, skip_channels, cond_channels,
                filter_width, activation, dilations, lpc_order, sample_after_filtering)
    {
    }


std::vector<at::Tensor> forward(
    const torch::Tensor &input,
    const torch::Tensor &a,
    const torch::Tensor &temperature
    )
{
    const int64_t batch_size = input.size(0);
    const int64_t timesteps = input.size(1);

    auto temperature_a = temperature.accessor<float, 3>();  


    auto output = torch::zeros({batch_size, timesteps, 1});

    // Accessors
    const auto input_a = input.accessor<float, 3>();
    auto output_a = output.accessor<float, 3>();
    auto a_a = a.accessor<float, 3>();

    // Process
    for (int64_t b = 0; b < batch_size; b++)
    {
        model.process(
            &input_a[b][0][0],
            &a_a[b][0][0],
            &temperature_a[b][0][0],
            &output_a[b][0][0],
            timesteps);
    }
    return {output};
}


std::vector<at::Tensor> cond_forward(
    const torch::Tensor &input,
    const torch::Tensor &a,
    const torch::Tensor &cond_input,
    const torch::Tensor &temperature
    )
{
    const int64_t batch_size = cond_input.size(0);
    const int64_t timesteps = cond_input.size(1);


    auto temperature_a = temperature.accessor<float, 3>();  

    auto output =  torch::zeros({batch_size, timesteps, 1});

    // Accessors
    const auto input_a = input.accessor<float, 3>();
    const auto cond_input_a = cond_input.accessor<float, 3>();
    auto output_a = output.accessor<float, 3>();
    auto a_a = a.accessor<float, 3>();

    // Process
    for (int64_t b = 0; b < batch_size; b++)
    {
        model.processConditional(
            &input_a[b][0][0],
            &a_a[b][0][0],
            &cond_input_a[b][0][0],
            &temperature_a[b][0][0],
            &output_a[b][0][0],
            timesteps);
    }
    return {output};
}

// Distribution persists across set_parameters / forward calls
// num_bits: categorical; components, shared_loc, discretize_bits: mixture
void setDistribution(const std::string &name, int num_bits,
                     const std::vector<std::string> &components,
                     bool shared_loc, int discretize_bits)
{
    distributions::DistributionSpec spec;
    spec.name = name;
    spec.num_bits = num_bits;
    spec.components = components;
    spec.shared_loc = shared_loc;
    spec.discretize_bits = discretize_bits;
    model.setDistribution(spec);
}

void setSeed(uint64_t seed)
{
    model.setSeed(seed);
}

void flush(int64_t timesteps)
{
    model.flush(timesteps);
}


void setParameters(const std::vector<torch::Tensor> &stack_weights_conv,
                    const std::vector<torch::Tensor> &stack_biases_conv,
                    const std::vector<torch::Tensor> &stack_weights_out,
                    const std::vector<torch::Tensor> &stack_biases_out,
                    const std::vector<torch::Tensor> &stack_weights_skip,
                    const std::vector<torch::Tensor> &stack_biases_skip,
                    const torch::Tensor &input_weight,
                    const torch::Tensor &input_bias,
                    const std::vector<torch::Tensor> &output_weights,
                    const std::vector<torch::Tensor> &output_biases)
{
    // Set parameters
    model.setInputWeight(glotnet::view(input_weight));
    model.setInputBias(glotnet::view(input_bias));
    int num_layers = dilations.size();
    for (size_t i = 0; i < num_layers; i++)
    {
        model.setStackConvolutionWeight(glotnet::view(stack_weights_conv[i]), i);
        model.setStackConvolutionBias(glotnet::view(stack_biases_conv[i]), i);
        model.setStackOutputWeight(glotnet::view(stack_weights_out[i]), i);
        model.setStackOutputBias(glotnet::view(stack_biases_out[i]), i);
        model.setStackSkipWeight(glotnet::view(stack_weights_skip[i]), i);
        model.setStackSkipBias(glotnet::view(stack_biases_skip[i]), i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        model.setOutputWeight(glotnet::view(output_weights[i]), i);
        model.setOutputBias(glotnet::view(output_biases[i]), i);
    }
    model.prepare();
}

void setParametersConditional(
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
    const std::vector<torch::Tensor> &output_biases)
{

    // Set parameters
    model.setInputWeight(glotnet::view(input_weight));
    model.setInputBias(glotnet::view(input_bias));
    int num_layers = dilations.size();
    for (size_t i = 0; i < num_layers; i++)
    {
        model.setStackConvolutionWeight(glotnet::view(stack_weights_conv[i]), i);
        model.setStackConvolutionBias(glotnet::view(stack_biases_conv[i]), i);
        model.setStackOutputWeight(glotnet::view(stack_weights_out[i]), i);
        model.setStackOutputBias(glotnet::view(stack_biases_out[i]), i);
        model.setStackSkipWeight(glotnet::view(stack_weights_skip[i]), i);
        model.setStackSkipBias(glotnet::view(stack_biases_skip[i]), i);
        model.setStackCondWeight(glotnet::view(stack_weights_cond[i]), i);
        model.setStackCondBias(glotnet::view(stack_biases_cond[i]), i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        model.setOutputWeight(glotnet::view(output_weights[i]), i);
        model.setOutputBias(glotnet::view(output_biases[i]), i);
    }

    model.prepare();
}

private:
    const int64_t input_channels;
    const int64_t output_channels;
    const int64_t convolution_channels;
    const int64_t skip_channels;
    const int64_t cond_channels;
    const int64_t filter_width;
    const int64_t lpc_order;
    const std::string activation;
    const std::vector<int> dilations;
    glotnet::GlotNetAR model;


};

} // namespace binding
} // namespace glotnet

void init_glotnet_ar(py::module &m)
{

    py::class_<glotnet::binding::GlotNetAR>(m, "GlotNetAR")
        .def(py::init<
            int,              // input_channels
            int,              // output_channels
            int,              // convolution_channels
            int,              // skip_channels
            int,              // cond_channels
            int,              // filter_width
            std::string,      // activation
            std::vector<int>, // dilations
            int,              // lpc_order
            bool              // sample_after_filtering
            >())
        .def("forward", &glotnet::binding::GlotNetAR::forward)
        .def("cond_forward", &glotnet::binding::GlotNetAR::cond_forward)
        .def("flush", &glotnet::binding::GlotNetAR::flush)
        .def("set_distribution", &glotnet::binding::GlotNetAR::setDistribution,
             py::arg("name"), py::arg("num_bits") = 8,
             py::arg("components") = std::vector<std::string>{},
             py::arg("shared_loc") = false, py::arg("discretize_bits") = 0)
        .def("set_seed", &glotnet::binding::GlotNetAR::setSeed)
        .def("set_parameters", &glotnet::binding::GlotNetAR::setParameters)
        .def("set_parameters_conditional", &glotnet::binding::GlotNetAR::setParametersConditional);
}