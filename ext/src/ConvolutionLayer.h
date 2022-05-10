
#pragma once

#include <vector>
#include <string>
#include <torch/extension.h>

#include "Convolution.h"
#include "Activations.h"

using namespace glotnet;
class ConvolutionLayer
{
public:
    ConvolutionLayer(size_t input_channels,
                     size_t output_channels,
                     size_t skip_channels,
                     size_t cond_channels,
                     size_t filter_width,
                     size_t dilation = 1,
                     bool use_output_transform = false,
                     std::string activation_name = "linear"); // TODO: use enum for activation types
    void process(const float *data_in, float *data_out, int64_t total_samples);
    void process(const float *data_in, float *data_out, float *skip_data, int64_t total_samples);
    void processConditional(const float *data_in, const float *conditioning, float *data_out, int64_t total_samples);
    void processConditional(const float *data_in, const float *conditioning, float *data_out, float *skip_data, int64_t total_samples);
    void reset();
    void prepare(size_t timesteps);
    void setConvolutionWeight(const torch::Tensor &W);
    void setConvolutionBias(const torch::Tensor &b);
    void setOutputWeight(const torch::Tensor &W);
    void setOutputBias(const torch::Tensor &b);
    void setSkipWeight(const torch::Tensor &W);
    void setSkipBias(const torch::Tensor &b);
    void setCondWeight(const torch::Tensor &W);
    void setCondBias(const torch::Tensor &b);

private:
    const size_t conv_out_channels;
    Convolution conv;
    Convolution out1x1;
    Convolution skip1x1;
    Convolution cond1x1;
    const bool use_output_transform;
    const bool use_gating;
    std::vector<float> memory;
    std::vector<float> memory_cond;
    typedef void (*activationFunction)(float *x, size_t rows, size_t cols); // TODO: make this into a class
    const activationFunction activation;                                          // TODO: make into a class
    inline void copyData(const float *data_src, int64_t channels_src, 
        float *data_dst, int64_t channels_dst, int64_t timesteps);
};
