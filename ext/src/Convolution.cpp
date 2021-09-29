/*
  ==============================================================================

    Convolution.cpp
    Author:  Damskägg Eero-Pekka, Lauri Juvela

  ==============================================================================
*/

#include "Convolution.h"

Convolution::Convolution(size_t inputChannels, size_t outputChannels, int filterWidth, int dilation) :
    bias(outputChannels),
    outVec(outputChannels),
    pos(0),
    dilation(dilation),
    inputChannels(inputChannels),
    outputChannels(outputChannels),
    filterWidth(filterWidth)
{
    resetFifo();
    resetKernel();
}

void Convolution::resetKernel()
{
    kernel.clear();
    kernel.reserve(filterWidth);
    for (int i = 0; i < filterWidth; ++i)
    {
        Eigen::MatrixXf x(inputChannels, outputChannels);
        x.setZero();
        kernel.push_back(x);
    }
    bias = Eigen::RowVectorXf(outputChannels);
    bias.setZero();
}

void Convolution::resetFifo()
{
    memory.clear();
    memory.reserve(getFilterOrder());
    for (int i = 0; i < getFilterOrder(); ++i)
    {
        Eigen::RowVectorXf x(inputChannels);
        x.setZero();
        memory.push_back(x);
    }
    pos = 0;
}

inline int Convolution::getFilterOrder() const
{
    return (filterWidth - 1) * dilation + 1;
}

void Convolution::process(const float *data_in, float *data_out, int64_t numSamples)
{
    for (int i = 0; i < numSamples; ++i)
    {
        processSingleSample(data_in, data_out, i, numSamples);
    }
}

void Convolution::processSingleSample(const float *data_in, float *data_out, int i, int numSamples)
{
    auto fifo = memory.begin();
    for (int ch = 0; ch < inputChannels; ++ch)
        (*(fifo + pos))[ch] = data_in[idx(ch, i, numSamples)];
    outVec.setZero();
    std::vector<Eigen::MatrixXf>::iterator it;
    int j = 0;
    for (auto it = kernel.begin(); it != kernel.end(); it++)
    {
        int readPos = mod((pos - j * dilation), getFilterOrder());
        outVec = outVec + *(fifo + readPos) * (*it);
        j += 1;
    }
    outVec = outVec + bias;
    for (int ch = 0; ch < outputChannels; ++ch)
        data_out[idx(ch, i, numSamples)] = outVec[ch];
    pos = mod(pos + 1, getFilterOrder());
}

void Convolution::processConditional(const float *data_in, const float *conditioning, float *data_out, int64_t numSamples)
{
    for (int i = 0; i < numSamples; ++i)
    {
        processSingleSampleConditional(data_in, conditioning, data_out, i, numSamples);
    }
}

void Convolution::processSingleSampleConditional(const float * data_in, const float * conditioning, float * data_out, int i, int numSamples)
{
    auto fifo = memory.begin();
    for (int ch = 0; ch < inputChannels; ++ch)
        (*(fifo + pos))[ch] = data_in[idx(ch, i, numSamples)];

    for (int ch = 0; ch < outputChannels; ++ch)
        outVec(ch) = conditioning[idx(ch, i, numSamples)];

    std::vector<Eigen::MatrixXf>::iterator it;
    int j = 0;
    for (auto it = kernel.begin(); it != kernel.end(); it++)
    {
        int readPos = mod((pos - j * dilation), getFilterOrder());
        outVec = outVec + *(fifo + readPos) * (*it);
        j += 1;
    }
    outVec = outVec + bias;
    for (int ch = 0; ch < outputChannels; ++ch)
        data_out[idx(ch, i, numSamples)] = outVec[ch];
    pos = mod(pos + 1, getFilterOrder());
}

int Convolution::mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

inline int64_t Convolution::idx(int64_t ch, int64_t i, int64_t numSamples)
{
    return ch * numSamples + i;
}

void Convolution::setKernel(const torch::Tensor &W)
{
    auto W_a = W.accessor<float, 3>(); // (out_channels, in_channels, kernel_size)
    assert(inputChannels == W.size(1));
    assert(outputChannels == W.size(0));
    assert(filterWidth == W.size(2));

    for (size_t k = 0; k < filterWidth; ++k)
        for (size_t row = 0; row < inputChannels; ++row)
            for (size_t col = 0; col < outputChannels; ++col)
                kernel[filterWidth - 1 - k](row, col) = W_a[col][row][k];
}

 void Convolution::setBias(const torch::Tensor &b)
 {
    auto b_a = b.accessor<float, 1>(); // (out_channels,)
    assert(b.size(0) == outputChannels);
    for (size_t i = 0; i < outputChannels; ++i)
        bias(i) = b_a[i];
 }