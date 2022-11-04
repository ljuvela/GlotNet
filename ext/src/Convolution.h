#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <torch/extension.h>
#include <vector>

namespace glotnet
{
class Convolution
{
public:
    Convolution(size_t input_channels, size_t output_channels, int filter_width, int dilation = 1);
    inline int getReceptiveField() const;
    void process(const float *data_in, float *data_out, int64_t total_samples);
    void processConditional(const float *data_in, const float *conditioning, float *data_out, int64_t total_samples);
    size_t getNumInputChannels() { return input_channels; }
    size_t getNumOutputChannels() { return output_channels; }
    void setWeight(const torch::Tensor &W);
    void setBias(const torch::Tensor &b);
    void setParameters(const std::vector<const torch::Tensor *> & params);
    void resetBuffer();
    void resetWeight();

private:
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> weight;
    Eigen::RowVectorXf bias;
    // TODO: benchmark against std::deque and boost circular buffer
    // TODO: https://eigen.tuxfamily.org/dox/group__TopicStlContainers.html
    // No workaround allocators needed post C++17 ?
    std::vector<Eigen::RowVectorXf, Eigen::aligned_allocator<Eigen::RowVectorXf>> memory;
    Eigen::RowVectorXf out_vec;
    int pos;
    const int dilation;
    const size_t input_channels;
    const size_t output_channels;
    const int filter_width;
    void processSingleSample(const float *data_in, float *data_out, int i, int total_samples);
    void processSingleSampleConditional(const float *data_in, const float *conditioning, float *data_out, int i, int total_samples);
    inline int mod(int a, int b) const;
};

class ConvolutionAR : public Convolution
{
public:
    ConvolutionAR(size_t input_channels, size_t output_channels, int filter_width, int dilation = 1);
    void process(const float *data_in, float *data_out, int64_t total_samples);

private:
    std::vector<float> x_curr;
    std::vector<float> x_prev;
};

} // glotnet
