#pragma once

#include <string>
#include "Activations.h"
#include "ConvolutionStack.h"
#include "WaveNet.h"

class WaveNetAR: public WaveNet
{
public:
    using WaveNet::WaveNet;
    void prepare();
    void process(float * const outputData, int total_samples);
    void processConditional(const float *conditioning, float * const output_data, int total_samples);

private:
    std::vector<float> input_buffer;
    std::vector<float> output_buffer;

};
