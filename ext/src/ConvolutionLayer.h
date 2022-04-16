
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
    ConvolutionLayer(size_t inputChannels,
                     size_t outputChannels,
                     int filterWidth,
                     int dilation = 1,
                     bool use_output_transform = false,
                     std::string activationName = "linear"); // TODO: use enum for activation types
    void process(const float *data_in, float *data_out, int64_t total_samples);
    void process(const float *data_in, float *data_out, float *skipdata, int64_t total_samples);
    void processConditional(const float *data_in, const float *conditioning, float *data_out, int64_t total_samples);
    void processConditional(const float *data_in, const float *conditioning, float *data_out, float *skipdata, int64_t total_samples);
    void reset();
    void setConvolutionWeight(const torch::Tensor &W);
    void setConvolutionBias(const torch::Tensor &b);
    void setOutputWeight(const torch::Tensor &W);
    void setOutputBias(const torch::Tensor &b);

private:
    Convolution conv;
    Convolution out1x1;
    bool use_output_transform;
    bool use_gating;
    std::vector<float> memory;
    void prepare(size_t num_channels, size_t buffer_size);
    typedef void (*activationFunction)(float *x, size_t rows, size_t cols); // TODO: make this into a class
    activationFunction activation;                                          // TODO: make into a class
    inline void copyData(const float *data_src, size_t channels_src, float *data_dst, size_t channels_dst, size_t timesteps);
};
