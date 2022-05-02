#pragma once

#include <string>
#include "Activations.h"
#include "ConvolutionStack.h"
class WaveNetAR
{
public:
    WaveNetAR(size_t input_channels, size_t output_channels, 
            size_t convolution_channels, size_t skip_channels, size_t cond_channels,
            size_t filter_width, std::string activation, std::vector<int> dilations);
    void prepare(int block_size);
    void process(float * const outputData, int total_samples);
    void processConditional(const float *conditioning, float * const outputData, int total_samples);
    void reset();
    void setStackConvolutionWeight(const torch::Tensor &W, int layerIdx);
    void setStackConvolutionBias(const torch::Tensor &b, int layerIdx);
    void setStackOutputWeight(const torch::Tensor &W, int layerIdx);
    void setStackOutputBias(const torch::Tensor &b, int layerIdx);
    void setInputWeight(const torch::Tensor &W);
    void setInputBias(const torch::Tensor &b);
    void setOutputWeight(const torch::Tensor &W, int layerIdx);
    void setOutputBias(const torch::Tensor &b, int layerIdx);

private:
    ConvolutionStack conv_stack;
    ConvolutionLayer input_layer;
    ConvolutionLayer output_layer1;
    ConvolutionLayer output_layer2;
    const int input_channels;
    const int output_channels;
    const int filter_width;
    const int skip_channels;
    const int convolution_channels;
    const int memory_channels;
    const std::string activation;
    const std::vector<int> dilations;
    static const int samples_per_block = 1;
    std::vector<float> conv_data;
    std::vector<float> skip_data;
    std::vector<float> inputBuffer;
    std::vector<float> outputBuffer;
    inline int idx(int ch, int i, int total_samples);
   
};
