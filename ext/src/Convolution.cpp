#include "Convolution.h"
#include <cassert>
namespace glotnet
{

Convolution::Convolution(size_t input_channels, size_t output_channels, int filter_width, int dilation) :
    bias(output_channels),
    out_vec(output_channels),
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
    for (auto & k : kernel)
    {
        const int read_pos = mod((pos - dilation * j++), receptive_field);
        out_vec += memory[read_pos] * k;
    }

    // copy output
    memcpy(&data_out[t * output_channels], out_vec.data(), sizeof(float) * output_channels);

    // update write pointer
    pos = mod(pos + 1, receptive_field);
}

void Convolution::processConditional(const float *data_in, const float *conditioning, float *data_out, int64_t total_samples)
{
    for (int i = 0; i < total_samples; ++i)
    {
        processSingleSampleConditional(data_in, conditioning, data_out, i, total_samples);
    }
}

void Convolution::processSingleSampleConditional(const float * data_in, const float * conditioning, 
float * data_out, int t, int total_samples)
{
    // update input buffer
    memcpy(memory[pos].data(), &data_in[t * input_channels], sizeof(float) * input_channels);

    // initialize output with conditioning
    memcpy(out_vec.data(), &conditioning[t * output_channels], sizeof(float) * output_channels);

    // dilated convolution
    int j = 0;
    const int receptive_field = getReceptiveField();
    for (auto & k : kernel)
    {
        const int read_pos = mod((pos - dilation * j++), receptive_field);
        out_vec += memory[read_pos] * k;
    }
    out_vec += bias;

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

void Convolution::setParameters(const std::vector<const torch::Tensor *> & params)
{
    assert (params.size() == 2);
    this->setKernel(*params[0]);
    this->setBias(*params[1]);
}

} // glotnet