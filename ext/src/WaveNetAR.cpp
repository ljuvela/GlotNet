#include "WaveNetAR.h"


WaveNetAR::WaveNetAR(int inputChannels, int outputChannels, int convolutionChannels,
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

void WaveNetAR::prepare(int buffer_size)
{
    convData.resize(samplesPerBlock * memoryChannels);
    skipData.resize(samplesPerBlock * skipChannels);
    inputBuffer.resize(inputChannels); // only single timestep
    outputBuffer.resize(outputChannels); // only single timestep
    convStack.prepare(samplesPerBlock);
    this->reset();
}

void WaveNetAR::reset()
{
    convStack.reset();
    inputLayer.reset();
    outputLayer1.reset();
    outputLayer2.reset();
}

void WaveNetAR::process(float * const outputData, int total_samples)
{
    std::fill(inputBuffer.begin(), inputBuffer.end(), 0.0f);
    for (int i = 0; i < total_samples; i++)
    {
        // calculate offsets

        std::cerr << "timestep " << i << "/" << total_samples << std::endl; 
        // NOTE data must be channels major

        // always process just one sample
        inputLayer.process(inputBuffer.data(), convData.data(), 1u);
        convStack.process(convData.data(), skipData.data(), 1u);
        outputLayer1.process(skipData.data(), convData.data(), 1u);
        outputLayer2.process(convData.data(), outputBuffer.data(), 1u);

        for (int t = 0; t < total_samples; t++)
        {
            std::cerr << outputData[t] << ", ";
        }
        std::cerr << std::endl;

        // copy output to input
        inputBuffer[0] = outputBuffer[0];
        outputData[i] = outputBuffer[0];

    }
}

void WaveNetAR::processConditional(const float *conditioning,
                                   float *const outputData, int total_samples)
{
    if (total_samples > samplesPerBlock)
        prepare(total_samples);
    inputLayer.process(inputBuffer.data(), convData.data(), total_samples);
    convStack.processConditional(convData.data(), conditioning, skipData.data(), total_samples);
    outputLayer1.process(skipData.data(), convData.data(), total_samples);
    outputLayer2.process(convData.data(), outputData, total_samples);
}

inline int WaveNetAR::idx(int ch, int i, int total_samples)
{
    return ch * total_samples + i;
}

void WaveNetAR::setStackConvolutionWeight(const torch::Tensor &W, int layerIdx)
{
    convStack.setConvolutionWeight(W, layerIdx);
}

void WaveNetAR::setStackConvolutionBias(const torch::Tensor &b, int layerIdx)
{
    convStack.setConvolutionBias(b, layerIdx);
}

void WaveNetAR::setStackOutputWeight(const torch::Tensor &W, int layerIdx)
{
    convStack.setOutputWeight(W, layerIdx);
}

void WaveNetAR::setStackOutputBias(const torch::Tensor &b, int layerIdx)
{
    convStack.setOutputBias(b, layerIdx);
}

void WaveNetAR::setInputWeight(const torch::Tensor &W)
{
    inputLayer.setConvolutionWeight(W);
}

void WaveNetAR::setInputBias(const torch::Tensor &b)
{
    inputLayer.setConvolutionBias(b);
}

void WaveNetAR::setOutputWeight(const torch::Tensor &W, int layerIdx)
{
    if (layerIdx == 0)
        outputLayer1.setConvolutionWeight(W);
    else if (layerIdx == 1)
        outputLayer2.setConvolutionWeight(W);
}

void WaveNetAR::setOutputBias(const torch::Tensor &b, int layerIdx)
{
    if (layerIdx == 0)
        outputLayer1.setConvolutionBias(b);
    else if (layerIdx == 1)
        outputLayer2.setConvolutionBias(b);
}
