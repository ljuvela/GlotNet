/*
  ==============================================================================

    ConvolutionLayer.h
    Created: 10 Jan 2019 5:04:39pm
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#pragma once

#include <iostream>

#include <vector>
#include <string>
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
  void process(const float *data_in, float *data_out, int64_t numSamples);
  void process(const float *data_in, float *data_out, float *skipdata, int64_t numSamples);
  void reset();
  void setConvolutionWeight(const float *data, size_t num_params);
  void setConvolutionBias(const float *data, size_t num_params);
  void setOutputWeight(const float *data, size_t num_params);
  void setOutputBias(const float *data, size_t num_params);

private:
  Convolution conv;
  Convolution out1x1;
  bool use_output_transform;
  bool use_gating;
  std::vector<float> memory;
  void prepare(size_t num_channels, size_t buffer_size);
  typedef void (*activationFunction)(float *x, size_t rows, size_t cols);
  activationFunction activation;
  void copySkipData(const float *data, float *skipData, int numSamples);
};
