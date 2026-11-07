#include "glotnet/Convolution.h"
#include <cassert>
#include <cstring>
namespace glotnet
{

Convolution::Convolution(size_t input_channels, size_t output_channels, int filter_width, int dilation) :
    bias(output_channels),
    out_vec(output_channels),
    cond_mul(output_channels),
    cond_add(output_channels),
    pos(0),
    dilation(dilation),
    input_channels(input_channels),
    output_channels(output_channels),
    filter_width(filter_width)
{
    resetBuffer();
    resetWeight();
}

void Convolution::resetWeight()
{
    weight.clear();
    weight.reserve(filter_width);
    for (int i = 0; i < filter_width; ++i)
    {
        Eigen::MatrixXf x(input_channels, output_channels);
        x.setZero();
        weight.push_back(x);
    }
    bias = Eigen::RowVectorXf(output_channels);
    bias.setZero();
}

void Convolution::resetBuffer()
{
    memory.clear();
    memory.reserve(getReceptiveField());
    for (int i = 0; i < getReceptiveField(); ++i)
    {
        Eigen::RowVectorXf x(input_channels);
        x.setZero();
        memory.push_back(x);
    }
    pos = 0;
}

inline int Convolution::getReceptiveField() const
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
    // update input buffer
    memcpy(memory[pos].data(), &data_in[t* input_channels], sizeof(float) * input_channels);

    // dilated convolution
    out_vec = bias;
    int j = 0;
    const size_t receptive_field = getReceptiveField();
    for (auto & w : weight)
    {
        const int read_pos = mod((pos - dilation * j++), receptive_field);
        out_vec += memory[read_pos] * w;
    }

    // copy output
    memcpy(&data_out[t * output_channels], out_vec.data(), sizeof(float) * output_channels);

    // update write pointer
    pos = mod(pos + 1, receptive_field);
}

void Convolution::processConditional(
    const float *data_in, const float *conditioning,
    float *data_out, int64_t total_samples, 
    const bool use_film)
{
    for (int i = 0; i < total_samples; ++i)
    {
        processSingleSampleConditional(
            data_in, conditioning,
            data_out, i, total_samples,
            use_film);
    }
}

void Convolution::processSingleSampleConditional(
    const float *data_in, const float *conditioning,
    float *data_out, int t, int total_samples, 
    const bool use_film)
{
    // update input buffer
    memcpy(memory[pos].data(), &data_in[t * input_channels], sizeof(float) * input_channels);

    if (use_film)
    {
        const unsigned int o = output_channels;
        memcpy(cond_add.data(), &conditioning[(t * 2) * o], sizeof(float) * output_channels);
        memcpy(cond_mul.data(), &conditioning[(t * 2) * o + o], sizeof(float) * output_channels);
    }
    else
    {
        memcpy(cond_add.data(), &conditioning[t * output_channels], sizeof(float) * output_channels);
    }

    // dilated convolution
    int j = 0;
    out_vec.setZero();
    const int receptive_field = getReceptiveField();
    for (auto & w : weight)
    {
        const int read_pos = mod((pos - dilation * j++), receptive_field);
        out_vec += memory[read_pos] * w;
    }
    if (use_film)
        out_vec = cond_mul.cwiseProduct(out_vec + bias) + cond_add;
    else
        out_vec += bias + cond_add;

    // copy output
    memcpy(&data_out[t * output_channels], out_vec.data(), sizeof(float) * output_channels);

    // update write pointer
    pos = mod(pos + 1, receptive_field);
}

inline int Convolution::mod(int a, int b) const
{
    const int r = a % b;
    const int rltz = r < 0;
    return rltz * (r + b) + (1 - rltz) * r;
}

void Convolution::setWeight(const ConstArrayView &W)
{
    const int64_t I = input_channels, O = output_channels, K = filter_width;
    W.checkShape({O, I, K}, "Convolution::setWeight");
    // W is (out_channels, in_channels, filter_width), row-major.
    // weight[k] is (in, out) and the taps are stored in reversed time order.
    for (int64_t k = 0; k < K; ++k)
        for (int64_t row = 0; row < I; ++row)
            for (int64_t col = 0; col < O; ++col)
                weight[K - 1 - k](row, col) = W.data[(col * I + row) * K + k];
}

void Convolution::setBias(const ConstArrayView &b)
{
    b.checkShape({static_cast<int64_t>(output_channels)}, "Convolution::setBias");
    for (size_t i = 0; i < output_channels; ++i)
        bias(i) = b.data[i];
}

ConvolutionAR::ConvolutionAR(size_t input_channels, size_t output_channels, int filter_width, int dilation)
: Convolution(input_channels, output_channels, filter_width, dilation)
{
    this->resetBuffer();
    x_curr.resize(1u * Convolution::getNumInputChannels()); 
    x_prev.resize(1u * Convolution::getNumInputChannels()); 
}


void ConvolutionAR::process(const float *data_in, float *data_out, int64_t total_samples)
{
    const size_t channels = Convolution::getNumInputChannels();
    
    // Timestep 0
    for (size_t c = 0; c < channels; c++)
    {
        data_out[0u + c] = data_in[0u + c];
        x_prev[c] = data_out[0u + c];
    }
    // Timesteps 1:total_samples
    for (int64_t t = 1; t < total_samples; t++)
    {
        Convolution::process(x_prev.data(), x_curr.data(), 1u);
        for (size_t c = 0; c < channels; c++)
        {
            data_out[t * channels + c] = x_curr[c] + data_in[t * channels + c];
            x_prev[c] = data_out[t  * channels + c];
        }
    }
}

} // glotnet