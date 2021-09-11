/*
  ==============================================================================

    WaveNet.h
    Created: 14 Jan 2019 5:19:01pm
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#pragma once

#include <string>
#include "Activations.h"
#include "ConvolutionStack.h"

class WaveNet
{
public:
    WaveNet(int inputChannels, int outputChannels, int convolutionChannels,
            int filterWidth, std::string activation, std::vector<int> dilations);
    void prepare(int block_size);
    void process(const float *inputData, float *outputData, int numSamples);
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
    ConvolutionStack convStack;
    ConvolutionLayer inputLayer;
    ConvolutionLayer outputLayer1;
    ConvolutionLayer outputLayer2;
    int inputChannels;
    int outputChannels;
    int filterWidth;
    int skipChannels;
    int convolutionChannels;
    int memoryChannels;
    std::string activation;
    std::vector<int> dilations;
    int samplesPerBlock = 0;
    std::vector<float> convData;
    std::vector<float> skipData;
    inline int idx(int ch, int i, int numSamples);
   
};
