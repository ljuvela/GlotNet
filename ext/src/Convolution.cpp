#include "Convolution.h"

namespace glotnet
{

Convolution::Convolution(size_t input_channels, size_t output_channels, int filter_width, int dilation) :
    bias(output_channels),
    outVec(output_channels),
    pos(0),
    dilation(dilation),
    input_channels(input_channels),
    output_channels(output_channels),
    filter_width(filter_width)
{
    resetFifo();
    resetKernel();
}

void Convolution::resetKernel()
{
    kernel.clear();
    kernel.reserve(filter_width);
    for (int i = 0; i < filter_width; ++i)
    {
        Eigen::MatrixXf x(input_channels, output_channels);
        x.setZero();
        kernel.push_back(x);
    }
    bias = Eigen::RowVectorXf(output_channels);
    bias.setZero();
}

void Convolution::resetFifo()
{
    memory.clear();
    memory.reserve(getFilterOrder());
    for (int i = 0; i < getFilterOrder(); ++i)
    {
        Eigen::RowVectorXf x(input_channels);
        x.setZero();
        memory.push_back(x);
    }
    pos = 0;
}

inline int Convolution::getFilterOrder() const
{
    return (filter_width - 1) * dilation + 1;
}

void Convolution::process(const float *data_in, float *data_out, int64_t total_samples)
{
    for (int i = 0; i < total_samples; ++i)
    {
        processSingleSample(data_in, data_out, i, total_samples);
    }
}

void Convolution::processSingleSample(const float *data_in, float *data_out, int t, int total_samples)
{
    auto fifo = memory.begin();
    for (size_t ch = 0; ch < input_channels; ++ch)
        fifo[pos][ch] = data_in[ch + t * input_channels];
    outVec = bias;
    int j = 0;
    for (auto & k : kernel)
    {
        const int readPos = mod((pos - dilation * j++), getFilterOrder());
        outVec = outVec + fifo[readPos] * k;
    }
    for (size_t ch = 0; ch < output_channels; ++ch)
        data_out[ch + t * output_channels] = outVec[ch];
    pos = mod(pos + 1, getFilterOrder());
}

void Convolution::processConditional(const float *data_in, const float *conditioning, float *data_out, int64_t total_samples)
{
    for (int i = 0; i < total_samples; ++i)
    {
        processSingleSampleConditional(data_in, conditioning, data_out, i, total_samples);
    }
}

void Convolution::processSingleSampleConditional(const float * data_in, const float * conditioning, float * data_out, int i, int total_samples)
{
    auto fifo = memory.begin();
    for (int ch = 0; ch < input_channels; ++ch)
        fifo[pos][ch] = data_in[idx_channel_major(ch, i, input_channels)];

    for (int ch = 0; ch < output_channels; ++ch)
        outVec(ch) = conditioning[idx_channel_major(ch, i, output_channels)];

    int j = 0;
    for (auto & k : kernel)
    {
        const int readPos = mod((pos - dilation * j++), getFilterOrder());
        outVec = outVec + fifo[readPos] * k;
    }
    outVec = outVec + bias;
    for (int ch = 0; ch < output_channels; ++ch)
        data_out[idx_channel_major(ch, i, output_channels)] = outVec[ch];
    pos = mod(pos + 1, getFilterOrder());
}

int Convolution::mod(int a, int b)
{
    int r = a % b;
    return r < 0 ? r + b : r;
}

inline int64_t Convolution::idx_time_major(int64_t c, int64_t t, int64_t total_samples)
{
    return c * total_samples + t;
}

inline int64_t Convolution::idx_channel_major(int64_t c, int64_t t, int64_t numChannels)
{
    return c + numChannels * t;
}

void Convolution::setKernel(const torch::Tensor &W)
{
    auto W_a = W.accessor<float, 3>(); // (out_channels, in_channels, kernel_size)
    assert(input_channels == W.size(1));
    assert(output_channels == W.size(0));
    assert(filter_width == W.size(2));

    for (size_t k = 0; k < filter_width; ++k)
        for (size_t row = 0; row < input_channels; ++row)
            for (size_t col = 0; col < output_channels; ++col)
                kernel[filter_width - 1 - k](row, col) = W_a[col][row][k];
}

void Convolution::setBias(const torch::Tensor &b)
{
    auto b_a = b.accessor<float, 1>(); // (out_channels,)
    assert(b.size(0) == output_channels);
    for (size_t i = 0; i < output_channels; ++i)
    bias(i) = b_a[i];
}

} // glotnet