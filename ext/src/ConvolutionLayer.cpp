/*
  ==============================================================================

    ConvolutionLayer.cpp
    Created: 10 Jan 2019 5:04:39pm
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer(size_t inputChannels,
                                   size_t outputChannels,
                                   int filterWidth,
                                   int dilation,
                                   bool use_output_transform,
                                   std::string activationName):
conv(inputChannels,
     Activations::isGated(activationName) ? outputChannels * 2 : outputChannels,
     filterWidth,
     dilation),
out1x1(outputChannels, outputChannels, 1, 1),
use_output_transform(use_output_transform),
use_gating(Activations::isGated(activationName)),
activation(Activations::getActivationFuncArray(activationName))
{
}

void ConvolutionLayer::process(const float * data_in, float * data_out, int64_t numSamples)
{
    this->prepare(conv.getNumOutputChannels(), numSamples);
    conv.process(data_in, memory.data(), numSamples);
    activation(memory.data(), conv.getNumOutputChannels(), numSamples);
    if (use_output_transform)
        out1x1.process(memory.data(), data_out, numSamples);
    else
        copySkipData(memory.data(), data_out, numSamples);
}

void ConvolutionLayer::process(const float * data_in,  float * data_out, float * skipData, int64_t numSamples)
{
    this->prepare(conv.getNumOutputChannels(), numSamples);
    conv.process(data_in, memory.data(), numSamples);
    activation(memory.data(), conv.getNumOutputChannels(), numSamples);
    copySkipData(memory.data(), skipData, numSamples);
    if (use_output_transform)
        out1x1.process(memory.data(), data_out, numSamples);
    else
        copySkipData(memory.data(), data_out, numSamples);
}

void ConvolutionLayer::copySkipData(const float *data, float *skipData, int numSamples)
{
    size_t skipChannels = use_gating ? conv.getNumOutputChannels()/2 : conv.getNumOutputChannels();
    for (size_t i = 0; i < (size_t)numSamples*skipChannels; ++i)
        skipData[i] = data[i];
}

void ConvolutionLayer::setConvolutionWeight(const torch::Tensor &W)
{
    conv.setKernel(W);
}

void ConvolutionLayer::setConvolutionBias(const torch::Tensor &b)
{
    conv.setBias(b);
}

void ConvolutionLayer::setOutputWeight(const torch::Tensor &W)
{
    out1x1.setKernel(W);
}

void ConvolutionLayer::setOutputBias(const torch::Tensor &b)
{
    out1x1.setBias(b);
}

void ConvolutionLayer::reset()
{
    conv.resetFifo();
    out1x1.resetFifo();
}

void ConvolutionLayer::prepare(size_t num_channels, size_t buffer_size)
{
    memory.resize(buffer_size * num_channels);
}