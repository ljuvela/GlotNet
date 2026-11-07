#pragma once
#include <string>
#include <vector>

#include "ConvolutionLayer.h"

class ConvolutionStack
{
public:
    ConvolutionStack(size_t num_channels, size_t num_skip_channels, size_t num_cond_channels,
                     size_t filter_width, std::vector<int> dilations,
                     std::string activation, bool residual = true);
    void process(float *data, float *skip_data, int total_samples);
    void processConditional(float *data, const float *conditioning, float *skip_data, int total_samples);
    void prepare(int buffer_size);
    void reset();
    size_t getNumLayers() { return dilations.size(); }
    void setConvolutionWeight(const glotnet::ConstArrayView &W, size_t layerIdx);
    void setConvolutionBias(const glotnet::ConstArrayView &b, size_t layerIdx);
    void setOutputWeight(const glotnet::ConstArrayView &W, size_t layerIdx);
    void setOutputBias(const glotnet::ConstArrayView &b, size_t layerIdx);
    void setSkipWeight(const glotnet::ConstArrayView &W, size_t layerIdx);
    void setSkipBias(const glotnet::ConstArrayView &b, size_t layerIdx);
    void setCondWeight(const glotnet::ConstArrayView &W, size_t layerIdx);
    void setCondBias(const glotnet::ConstArrayView &b, size_t layerIdx);

private:
    std::vector<ConvolutionLayer> layers;
    const std::vector<int> dilations;
    const size_t num_layers;
    const bool residual;
    const size_t num_channels;
    const size_t num_cond_channels;
    const size_t num_skip_channels;
    const size_t filter_width;
    const std::string activation;
    int samples_per_block = 0;
    std::vector<float> residual_data;
    void saveResidual(const float *data, int total_samples);
    void addResidual(float *data, int total_samples);
    void initLayers();
};
