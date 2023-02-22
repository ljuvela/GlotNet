#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/GlotNetAR.h"

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
              size_t lpc_order)
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
                filter_width, activation, dilations, lpc_order)
    {
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
    model.setInputWeight(input_weight);
    model.setInputBias(input_bias);
    int num_layers = dilations.size();
    for (size_t i = 0; i < num_layers; i++)
    {
        model.setStackConvolutionWeight(stack_weights_conv[i], i);
        model.setStackConvolutionBias(stack_biases_conv[i], i);
        model.setStackOutputWeight(stack_weights_out[i], i);
        model.setStackOutputBias(stack_biases_out[i], i);
        model.setStackSkipWeight(stack_weights_skip[i], i);
        model.setStackSkipBias(stack_biases_skip[i], i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        model.setOutputWeight(output_weights[i], i);
        model.setOutputBias(output_biases[i], i);
    }
}

std::vector<at::Tensor> forward(
    const torch::Tensor &input,
    const torch::Tensor &a,
    bool use_residual=true,
    float temperature=1.0
    )
{
    const int64_t batch_size = input.size(0);
    const int64_t timesteps = input.size(1);

    model.prepare();
    model.setDistribution("gaussian");
    model.setSamplingTemperature(temperature);

    auto output = torch::zeros({batch_size, timesteps, 1});

    // Accessors
    const auto input_a = input.accessor<float, 3>();
    auto output_a = output.accessor<float, 3>();
    auto a_a = a.accessor<float, 3>();

    // Process
    for (int64_t b = 0; b < batch_size; b++)
    {
        model.reset();
        model.process(
            &input_a[b][0][0],
            &a_a[b][0][0],
            &output_a[b][0][0],
            timesteps);
    }
    return {output};
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
    model.setInputWeight(input_weight);
    model.setInputBias(input_bias);
    int num_layers = dilations.size();
    for (size_t i = 0; i < num_layers; i++)
    {
        model.setStackConvolutionWeight(stack_weights_conv[i], i);
        model.setStackConvolutionBias(stack_biases_conv[i], i);
        model.setStackOutputWeight(stack_weights_out[i], i);
        model.setStackOutputBias(stack_biases_out[i], i);
        model.setStackSkipWeight(stack_weights_skip[i], i);
        model.setStackSkipBias(stack_biases_skip[i], i);
        model.setStackCondWeight(stack_weights_cond[i], i);
        model.setStackCondBias(stack_biases_cond[i], i);
    }
    for (size_t i = 0; i < output_weights.size(); i++)
    {
        model.setOutputWeight(output_weights[i], i);
        model.setOutputBias(output_biases[i], i);
    }
}

std::vector<at::Tensor> cond_forward(
    const torch::Tensor &input,
    const torch::Tensor &a,
    const torch::Tensor &cond_input,
    bool use_residual=true,
    float temperature=1.0
    )
{
    const int64_t batch_size = cond_input.size(0);
    const int64_t timesteps = cond_input.size(1);
    const int64_t cond_channels = cond_input.size(2);

    // Set buffer size to match timesteps
    model.prepare();
    model.setDistribution("gaussian");
    model.setSamplingTemperature(temperature);

    auto output =  torch::zeros({batch_size, timesteps, 1});

    // Accessors
    const auto input_a = input.accessor<float, 3>();
    const auto cond_input_a = cond_input.accessor<float, 3>();
    auto output_a = output.accessor<float, 3>();
    auto a_a = a.accessor<float, 3>();

    // Process
    for (int64_t b = 0; b < batch_size; b++)
    {
        model.reset();
        model.processConditional(
            &input_a[b][0][0],
            &a_a[b][0][0],
            &cond_input_a[b][0][0],
            &output_a[b][0][0],
            timesteps);
    }
    return {output};
}

void flush(int64_t timesteps)
{
    model.flush(timesteps);
}



private:

glotnet::GlotNetAR model;

const int64_t input_channels;
const int64_t output_channels;
const int64_t convolution_channels;
const int64_t skip_channels;
const int64_t cond_channels;
const int64_t filter_width;
const int64_t lpc_order;
const std::string activation;
const std::vector<int> dilations;


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
         int               // lpc_order
         >())
    .def("forward", &glotnet::binding::GlotNetAR::forward)
    .def("cond_forward", &glotnet::binding::GlotNetAR::cond_forward)
    .def("flush", &glotnet::binding::GlotNetAR::flush)
    .def("set_parameters", &glotnet::binding::GlotNetAR::setParameters)
    .def("set_parameters_conditional", &glotnet::binding::GlotNetAR::setParametersConditional);
}