/*
  ==============================================================================

    ConvolutionLayer.h
    Created: 10 Jan 2019 5:04:39pm
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#pragma once

#include <vector>
#include <string>
#include <torch/extension.h>

#include "Convolution.h"
#include "Activations.h"

class ConvolutionLayer
{
public:
  ConvolutionLayer(size_t inputChannels,
                   size_t outputChannels,
                   int filterWidth,
                   int dilation = 1,
                   bool use_output_transform = false,
                   std::string activationName = "linear");
  void process(const float *data_in, float *data_out, int64_t total_samples);
  void process(const float *data_in, float *data_out, float *skipdata, int64_t total_samples);
  void processConditional(const float *data_in, const float *conditioning, float *data_out, int64_t total_samples);
  void processConditional(const float *data_in, const float *conditioning, float *data_out, float *skipdata, int64_t total_samples);
  void reset();
  void setConvolutionWeight(const torch::Tensor &W);
  void setConvolutionBias(const torch::Tensor &b);
  void setOutputWeight(const torch::Tensor &W);
  void setOutputBias(const torch::Tensor &b);

private:
  Convolution conv;
  Convolution out1x1;
  bool use_output_transform;
  bool use_gating;
  std::vector<float> memory;
  void prepare(size_t num_channels, size_t buffer_size);
  typedef void (*activationFunction)(float *x, size_t rows, size_t cols);
  activationFunction activation;
  void copySkipData(const float *data, float *skipData, int total_samples);
};
