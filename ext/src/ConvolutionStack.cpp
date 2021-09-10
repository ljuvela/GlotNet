/*
  ==============================================================================

    ConvolutionStack.cpp
    Created: 8 Jan 2019 5:21:49pm
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#include "ConvolutionStack.h"

ConvolutionStack::ConvolutionStack(int numChannels, int filterWidth, std::vector<int> dilations, std::string activation, bool residual) :
    dilations(dilations),
    residual(residual),
    numChannels(numChannels),
    filterWidth(filterWidth),
    activation(activation)
{
    initLayers();
}

void ConvolutionStack::prepare(int buffer_size)
{
    samplesPerBlock = buffer_size;
    residualData.resize(samplesPerBlock * numChannels);
}

void ConvolutionStack::reset()
{
    for (auto l : layers)
        l.reset();
    std::fill(residualData.begin(), residualData.end(), 0.0f);
}

void ConvolutionStack::copyResidual(const float *data, int numSamples)
{
    for (size_t i = 0; i < numSamples * numChannels; ++i)
        residualData[i] = data[i];
}

void ConvolutionStack::addResidual(float *data, int numSamples)
{
    for (size_t i = 0; i < numSamples * numChannels; ++i)
        data[i] = data[i] + residualData[i];
}

void ConvolutionStack::process(float *data, float* skipData, int numSamples)
{
    if (numSamples > samplesPerBlock)
        prepare(numSamples);
    for (int i = 0; i < dilations.size(); ++i)
    {
        if (residual)
            copyResidual(data, numSamples);
        // Get pointer to correct position at skipData
        float *skipPtr = getSkipPointer(skipData, i, numSamples);
        layers[i].process(data, data, skipPtr, numSamples);
        if (residual)
            addResidual(data, numSamples);
    }
}

float* ConvolutionStack::getSkipPointer(float *skipData, int layerIdx, int numSamples)
{
    int startCh = numChannels * layerIdx;
    int startIdx = idx(startCh, 0, numSamples);
    return &skipData[startIdx];
}

inline unsigned int ConvolutionStack::idx(int ch, int i, int numSamples)
{
    return ch * numSamples + i;
}

void ConvolutionStack::setConvolutionWeight(const torch::Tensor &W, size_t layerIdx)
{
    layers[layerIdx].setConvolutionWeight(W);
}

void ConvolutionStack::setConvolutionBias(const torch::Tensor &b, size_t layerIdx)
{
    layers[layerIdx].setConvolutionBias(b);
}

void ConvolutionStack::setOutputWeight(const torch::Tensor &W, size_t layerIdx)
{
    layers[layerIdx].setOutputWeight(W);
}

void ConvolutionStack::setOutputBias(const torch::Tensor &b, size_t layerIdx)
{
    layers[layerIdx].setOutputBias(b);
}

void ConvolutionStack::initLayers()
{
    layers.clear();
    layers.reserve(dilations.size());
    for (size_t i = 0; i < dilations.size(); ++i)
    {
        bool use_output_transform = true;
        if (i == dilations.size() - 1)
            use_output_transform = false;
        layers.push_back(ConvolutionLayer(numChannels,
                                          numChannels,
                                          filterWidth,
                                          dilations[i],
                                          use_output_transform,
                                          activation));
    }

}
