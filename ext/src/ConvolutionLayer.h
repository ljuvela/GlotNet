/*
  ==============================================================================

    ConvolutionLayer.h
    Created: 10 Jan 2019 5:04:39pm
    Author:  Damskägg Eero-Pekka

  ==============================================================================
*/

#pragma once


#include <iostream>

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
                     bool residual = false,
                     std::string activationName = "linear");
    ~ConvolutionLayer(){std::cerr << "Deleting layer" << std::endl;};
    void process(float * data_in, float * data_out, int numSamples);
    void process(float* data_in, float * data_out, float* skipdata, int numSamples);
    void setConvolutionWeight(float * data, size_t num_params);
    void setConvolutionBias(float * data, size_t num_params);
    void setOutputWeight(float * data, size_t num_params);
    void setOutputBias(float * data, size_t num_params);

    
private:
    Convolution conv;
    Convolution out1x1;
    bool residual;
    bool usesGating;
    typedef void (* activationFunction)(float *x , size_t rows, size_t cols);
    activationFunction activation;
    void copySkipData(float *data, float *skipData, int numSamples);
};
