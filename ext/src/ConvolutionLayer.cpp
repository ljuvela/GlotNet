#include "ConvolutionLayer.h"

ConvolutionLayer::ConvolutionLayer(
    size_t input_channels,
    size_t output_channels,
    size_t skip_channels,
    size_t cond_channels,
    size_t filter_width,
    size_t dilation,
    bool use_output_transform,
    std::string activation_name)
    : conv_out_channels(Activations::isGated(activation_name) ? output_channels * 2 : output_channels),
      conv(input_channels, conv_out_channels, filter_width, dilation),
      out1x1(output_channels, output_channels, 1, 1),
      skip1x1(output_channels, skip_channels, 1, 1),
      cond1x1(cond_channels, conv_out_channels, 1, 1),
      use_output_transform(use_output_transform),
      use_gating(Activations::isGated(activation_name)),
      activation(Activations::getActivationFuncArray(activation_name))
{
}

void ConvolutionLayer::process(const float *data_in, float *data_out, int64_t timesteps)
{
    const size_t conv_out_ch = conv.getNumOutputChannels();
    const size_t out_ch = use_gating ? conv_out_ch / 2 : conv_out_ch;
    this->prepare(timesteps);
    conv.process(data_in, memory.data(), timesteps);
    activation(memory.data(), timesteps, conv_out_ch);
    if (use_output_transform)
        out1x1.process(memory.data(), data_out, timesteps);
    else
        copyData(memory.data(), out_ch, data_out, out_ch, timesteps);
}

void ConvolutionLayer::process(
    const float *data_in, float *data_out,
    float *skip_data, int64_t timesteps)
{
    const size_t conv_out_ch = conv.getNumOutputChannels();
    const size_t out_ch = use_gating ? conv_out_ch / 2 : conv_out_ch;
    this->prepare(timesteps); // TODO: take prepare call out of process loop
    conv.process(data_in, memory.data(), timesteps);
    activation(memory.data(), timesteps, conv_out_ch);
    copyData(memory.data(), conv_out_ch, data_out, out_ch, timesteps);
    skip1x1.process(data_out, skip_data, timesteps);
    if (use_output_transform)
        out1x1.process(data_out, data_out, timesteps);

}

void ConvolutionLayer::processConditional(
    const float *data_in, const float *conditioning,
    float *data_out, float *skip_data, int64_t timesteps)
{
    const size_t conv_out_ch = conv.getNumOutputChannels();
    const size_t out_ch = use_gating ? conv_out_ch / 2 : conv_out_ch;
    this->prepare(timesteps); // TODO: take prepare call out of process loop
    cond1x1.process(conditioning, memory_cond.data(), timesteps);
    conv.processConditional(data_in, memory_cond.data(), memory.data(), timesteps);
    activation(memory.data(), timesteps, conv_out_ch);
    copyData(memory.data(), conv_out_ch, data_out, out_ch, timesteps);
    skip1x1.process(data_out, skip_data, timesteps);
    if (use_output_transform)
        out1x1.process(data_out, data_out, timesteps);
}

inline void ConvolutionLayer::copyData(const float *data_src, int64_t ch_src,
                                float *data_dst, int64_t ch_dst, int64_t timesteps)
{
    for (size_t t = 0; t < timesteps; t++)
        memcpy(&data_dst[t * ch_dst], &data_src[t * ch_src], ch_dst * sizeof(float));
}

void ConvolutionLayer::setConvolutionWeight(const torch::Tensor &W)
{
    conv.setWeight(W);
}

void ConvolutionLayer::setConvolutionBias(const torch::Tensor &b)
{
    conv.setBias(b);
}

void ConvolutionLayer::setOutputWeight(const torch::Tensor &W)
{
    out1x1.setWeight(W);
}

void ConvolutionLayer::setOutputBias(const torch::Tensor &b)
{
    out1x1.setBias(b);
}

void ConvolutionLayer::setSkipWeight(const torch::Tensor &W)
{
    skip1x1.setWeight(W);
}

void ConvolutionLayer::setSkipBias(const torch::Tensor &b)
{
    skip1x1.setBias(b);
}

void ConvolutionLayer::setCondWeight(const torch::Tensor &W)
{
    cond1x1.setWeight(W);
}

void ConvolutionLayer::setCondBias(const torch::Tensor &b)
{
    cond1x1.setBias(b);
}

void ConvolutionLayer::reset()
{
    conv.resetBuffer();
    out1x1.resetBuffer();
    skip1x1.resetBuffer();
    cond1x1.resetBuffer();
}

void ConvolutionLayer::prepare(size_t timesteps)
{
    const size_t conv_out_ch = conv.getNumOutputChannels();
    memory.resize(timesteps * conv_out_ch);
    memory_cond.resize(timesteps * conv_out_ch);
}