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

void ConvolutionStack::copyResidual(const float *data, int total_samples)
{
    for (size_t i = 0; i < total_samples * numChannels; ++i)
        residualData[i] = data[i];
}

void ConvolutionStack::addResidual(float *data, int total_samples)
{
    for (size_t i = 0; i < total_samples * numChannels; ++i)
        data[i] = data[i] + residualData[i];
}

void ConvolutionStack::process(float *data, float* skipData, int total_samples)
{
    if (total_samples > samplesPerBlock)
        prepare(total_samples);
    for (int i = 0; i < dilations.size(); ++i)
    {
        if (residual)
            copyResidual(data, total_samples);
        // Get pointer to correct position at skipData
        float *skipPtr = getSkipPointer(skipData, i, total_samples);
        layers[i].process(data, data, skipPtr, total_samples);
        if (residual)
            addResidual(data, total_samples);
    }
}

void ConvolutionStack::processConditional(float *data, const float *conditioning, float* skipData, int total_samples)
{
    if (total_samples > samplesPerBlock)
        prepare(total_samples);
    for (int i = 0; i < dilations.size(); ++i)
    {
        if (residual)
            copyResidual(data, total_samples);
        // Get pointer to correct position at skipData
        float *skipPtr = getSkipPointer(skipData, i, total_samples);
        const float *condPtr = getCondPointer(conditioning, i, total_samples);
        layers[i].processConditional(data, condPtr,
                                     data, skipPtr, total_samples);
        if (residual)
            addResidual(data, total_samples);
    }
}

float* ConvolutionStack::getSkipPointer(float *data, int layerIdx, int total_samples)
{
    const int startCh = numChannels * layerIdx;
    const int startIdx = idx(startCh, 0, total_samples);
    return &data[startIdx];
}

const float* ConvolutionStack::getCondPointer(const float *data, int layerIdx, int total_samples)
{
    // conditioning has twice the number of residual channels (one for each gate)
    const int startCh = 2 * numChannels * layerIdx;
    const int startIdx = idx(startCh, 0, total_samples);
    return &data[startIdx];
}

inline unsigned int ConvolutionStack::idx(int ch, int i, int total_samples)
{
    return ch * total_samples + i;
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
