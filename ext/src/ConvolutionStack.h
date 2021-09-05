/*
  ==============================================================================

    ConvolutionStack.h
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#pragma once

#include "ConvolutionLayer.h"

class ConvolutionStack
{
public:
    ConvolutionStack(int numChannels, int filterWidth, std::vector<int> dilations,
                     std::string activation, bool residual = true);
    void process(float * data, float * skipData, int numSamples);
    void prepare(int buffer_size);
    void reset();
    size_t getNumLayers() { return dilations.size(); }
    void setConvolutionWeight(const float * data, size_t layerIdx, size_t num_params);
    void setConvolutionBias(const float * data, size_t layerIdx, size_t num_params);
    void setOutputWeight(const float * data, size_t layerIdx, size_t num_params);
    void setOutputBias(const float * data, size_t layerIdx, size_t num_params);
    
private:
    std::vector<ConvolutionLayer> layers;
    std::vector<int> dilations;
    bool residual;
    int numChannels;
    int filterWidth;
    std::string activation;
    int samplesPerBlock = 0;
    std::vector<float> residualData;
    inline unsigned int idx(int ch, int i, int numSamples);
    void copyResidual(const float *data, int numSamples);
    void addResidual(float *data, int numSamples);
    float* getSkipPointer(float *skipData, int layerIdx, int numSamples);
    void initLayers();
};
