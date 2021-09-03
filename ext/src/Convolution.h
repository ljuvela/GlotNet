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

class Convolution
{
public:
  Convolution(size_t inputChannels, size_t outputChannels, int filterWidth, int dilation = 1);
  int getFilterOrder() const;
  void process(float * data_in, float * data_out, int numSamples);
  size_t getNumInputChannels() { return inputChannels; }
  size_t getNumOutputChannels() { return outputChannels; }
  void setKernel(float *W, size_t num_params);
  void setBias(float *b, size_t num_params);
  void resetFifo();

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

  void resetKernel();
  void processSingleSample(float * data_in, float * data_out, int i, int numSamples);

  int mod(int a, int b);
  int idx(int ch, int i, int numSamples);
};
