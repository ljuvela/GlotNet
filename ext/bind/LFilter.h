#include <torch/extension.h>

#include <vector>

// #include "../src/LFilter.h"


namespace glotnet
{

using namespace torch::indexing;

class LFilter
{

public:

    void forward(const torch::Tensor & input, const torch::Tensor & a, const torch::Tensor & b, torch::Tensor & output)
    {

        c10::InferenceMode guard; // disable autograd for higher efficiency

        int64_t batch = input.size(0);
        int64_t timesteps = input.size(1);
        int64_t channels = input.size(2);

        int64_t order = a.size(2)-1;

        resize(batch, channels, order);

        // normalize a
        auto a_norm = a / a.index({"...", Slice(0,1)});

        auto input_a = input.accessor<float, 3>();   // size (batch, time, channels)
        auto output_a = output.accessor<float, 3>(); // size (batch, time, channels)
        auto a_a = a.accessor<float, 3>(); // size (batch, time, order+1)
        auto b_a = b.accessor<float, 3>(); // size (batch, time, order+1)
        auto state_a = state.accessor<float, 3>(); // size (batch, channels, order)

        for (int64_t b = 0; b < batch; b++)
        {
            for (int64_t c = 0; c < channels; c++)
            {
                for (int64_t t = 0; t < timesteps; t++)
                {
                    float y = 0;
                    // feedforward
                    for (int64_t i = 0; i < order + 1; i++)
                    {
                        const int idx = t-i;
                        const unsigned int mask = idx >= 0;
                        y += b_a[b][t][i] * input_a[b][idx * mask][c] * mask;
                    }
                    // feedback
                    for (int64_t i = 0; i < order; i++)
                    {
                        y -= a_a[b][t][i+1] * state_a[b][c][i];
                    }

                    output_a[b][t][c] = y;
                    // update state
                    for (int64_t i = order; i > 0; i--)
                    {
                        state_a[b][c][i] = state_a[b][c][i-1];
                    }
                    state_a[b][c][0] = y;
                }
            }
        }
    }



private:

    void resize(int64_t batch, int64_t channels, int64_t order)
    {
        if (batch != state.size(0) || channels != state.size(1) || order != state.size(2))
        {
            state = torch::zeros({batch, channels, order});
        }
    }

    torch::Tensor state; // size (batch, channels, order)


};



} // glotnet

void init_lfilter(py::module &m)
{
    py::class_<glotnet::LFilter>(m, "LFilter")
        .def(py::init<>())
        .def("forward", &glotnet::LFilter::forward);
}