// Uses GlotNet like a plugin would: construct a model, load weights from a
// plain float buffer (torch layout), prepare for a block size, process.
#include <cmath>
#include <cstdio>
#include <vector>

#include "glotnet/WaveNet.h"

int main()
{
    const int block_size = 64;
    const int64_t channels = 4, filter_width = 2;

    glotnet::WaveNet wavenet(1, 2, channels, channels, 0, filter_width,
                             "gated", {1, 2, 4, 8});

    // Input layer: 1x1 convolution from 1 to channels inputs
    std::vector<float> w_in(channels * 1 * 1, 0.1f), b_in(channels, 0.0f);
    wavenet.setInputWeight({w_in.data(), {channels, 1, 1}});
    wavenet.setInputBias({b_in.data(), {channels}});

    wavenet.prepare(block_size);
    std::vector<float> in(block_size, 0.25f), out(block_size * 2);
    wavenet.process(in.data(), out.data(), block_size);

    for (float v : out)
        if (!std::isfinite(v))
        {
            std::fprintf(stderr, "non-finite output\n");
            return 1;
        }
    std::printf("consumer ok: out[0] = %g\n", out[0]);
    return 0;
}
