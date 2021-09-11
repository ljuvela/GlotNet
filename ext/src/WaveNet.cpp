/*
  ==============================================================================

    WaveNet.cpp
    Created: 14 Jan 2019 5:19:01pm
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#include "WaveNet.h"


WaveNet::WaveNet(int inputChannels, int outputChannels, int convolutionChannels,
                 int filterWidth, std::string activation, std::vector<int> dilations) :
inputChannels(inputChannels),
outputChannels(outputChannels),
filterWidth(filterWidth),
skipChannels(convolutionChannels * (int)dilations.size()),
convStack(convolutionChannels, filterWidth, dilations, activation),
inputLayer(inputChannels, convolutionChannels, 1, 1, false, "tanh"),
outputLayer1(convolutionChannels * dilations.size(), convolutionChannels, 1, 1, false, "tanh"),
outputLayer2(convolutionChannels, outputChannels, 1, 1, false, "linear"),
convolutionChannels(convolutionChannels),
memoryChannels(Activations::isGated(activation) ? convolutionChannels * 2 : convolutionChannels),
activation(activation),
dilations(dilations)
{
}

void WaveNet::prepare(int buffer_size)
{
    samplesPerBlock = buffer_size;
    convData.resize(samplesPerBlock * memoryChannels);
    skipData.resize(samplesPerBlock * skipChannels);
    convStack.prepare(samplesPerBlock);
    this->reset();
}

void WaveNet::reset()
{
    convStack.reset();
    inputLayer.reset();
    outputLayer1.reset();
    outputLayer2.reset();
}

void WaveNet::process(const float *inputData, float *outputData, int numSamples)
{
    if (numSamples > samplesPerBlock)
        prepare(numSamples);
    inputLayer.process(inputData, convData.data(), numSamples);
    convStack.process(convData.data(), skipData.data(), numSamples);

    outputLayer1.process(skipData.data(), convData.data(), numSamples);
    outputLayer2.process(convData.data(), outputData, numSamples);

}

inline int WaveNet::idx(int ch, int i, int numSamples)
{
    return ch * numSamples + i;
}

void WaveNet::setStackConvolutionWeight(const torch::Tensor &W, int layerIdx)
{
    convStack.setConvolutionWeight(W, layerIdx);
}

void WaveNet::setStackConvolutionBias(const torch::Tensor &b, int layerIdx)
{
    convStack.setConvolutionBias(b, layerIdx);
}

void WaveNet::setStackOutputWeight(const torch::Tensor &W, int layerIdx)
{
    convStack.setOutputWeight(W, layerIdx);
}

void WaveNet::setStackOutputBias(const torch::Tensor &b, int layerIdx)
{
    convStack.setOutputBias(b, layerIdx);
}

void WaveNet::setInputWeight(const torch::Tensor &W)
{
    inputLayer.setConvolutionWeight(W);
}

void WaveNet::setInputBias(const torch::Tensor &b)
{
    inputLayer.setConvolutionBias(b);
}

void WaveNet::setOutputWeight(const torch::Tensor &W, int layerIdx)
{
    if (layerIdx == 0)
        outputLayer1.setConvolutionWeight(W);
    else if (layerIdx == 1)
        outputLayer2.setConvolutionWeight(W);
}

void WaveNet::setOutputBias(const torch::Tensor &b, int layerIdx)
{
    if (layerIdx == 0)
        outputLayer1.setConvolutionBias(b);
    else if (layerIdx == 1)
        outputLayer2.setConvolutionBias(b);
}
