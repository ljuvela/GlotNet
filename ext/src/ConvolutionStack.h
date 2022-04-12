/*
  ==============================================================================

    ConvolutionStack.h
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#pragma once
#include <torch/extension.h>

#include "ConvolutionLayer.h"

class ConvolutionStack
{
public:
    ConvolutionStack(int numChannels, int filterWidth, std::vector<int> dilations,
                     std::string activation, bool residual = true);
    void process(float * data, float * skipData, int total_samples);
    void processConditional(float * data, const float *conditioning, float * skipData, int total_samples);
    void prepare(int buffer_size);
    void reset();
    size_t getNumLayers() { return dilations.size(); }
    void setConvolutionWeight(const torch::Tensor &W, size_t layerIdx);
    void setConvolutionBias(const torch::Tensor &b, size_t layerIdx);
    void setOutputWeight(const torch::Tensor &W, size_t layerIdx);
    void setOutputBias(const torch::Tensor &b, size_t layerIdx);

private:
    std::vector<ConvolutionLayer> layers;
    std::vector<int> dilations;
    bool residual;
    int numChannels;
    int filterWidth;
    std::string activation;
    int samplesPerBlock = 0;
    std::vector<float> residualData;
    inline unsigned int idx(int ch, int i, int total_samples);
    void copyResidual(const float *data, int total_samples);
    void addResidual(float *data, int total_samples);
    float* getSkipPointer(float *skipData, int layerIdx, int total_samples);
    const float* getCondPointer(const float *data, int layerIdx, int total_samples);
    void initLayers();
};
