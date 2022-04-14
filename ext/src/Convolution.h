/*
  ==============================================================================

    Convolution.h
    Created: 3 Jan 2019 10:58:34am
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#pragma once

#include <Eigen/Core>
#include <Eigen/StdVector>
#include <torch/extension.h>

namespace glotnet
{

class Convolution
{
public:
    Convolution(size_t inputChannels, size_t outputChannels, int filterWidth, int dilation = 1);
    inline int getFilterOrder() const;
    void process(const float *data_in, float *data_out, int64_t total_samples);
    void processConditional(const float *data_in, const float *conditioning, float *data_out, int64_t total_samples);
    size_t getNumInputChannels() { return inputChannels; }
    size_t getNumOutputChannels() { return outputChannels; }
    void setKernel(const torch::Tensor &W);
    void setBias(const torch::Tensor &b);
    void resetFifo();
    void resetKernel();

    static inline int64_t idx_time_major(int64_t c, int64_t t, int64_t total_samples);
    static inline int64_t idx_channel_major(int64_t c, int64_t t, int64_t numChannels);

private:
    std::vector<Eigen::MatrixXf, Eigen::aligned_allocator<Eigen::MatrixXf>> kernel;
    Eigen::RowVectorXf bias;
    std::vector<Eigen::RowVectorXf, Eigen::aligned_allocator<Eigen::RowVectorXf>> memory;
    Eigen::RowVectorXf outVec;
    int pos;
    const int dilation;
    const size_t inputChannels;
    const size_t outputChannels;
    const int filterWidth;
    void processSingleSample(const float *data_in, float *data_out, int i, int total_samples);
    void processSingleSampleConditional(const float *data_in, const float *conditioning, float *data_out, int i, int total_samples);
    int mod(int a, int b);

};

} // glotnet
