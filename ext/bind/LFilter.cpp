#include <torch/extension.h>

#include <vector>



#include "../src/LFilter.h"


namespace glotnet
{

class LFilter
{

public:

    forward(const torch::Tensor & input, const torch::Tensor & a, const torch::Tensor & b, torch::Tensor & output)
    {
        int64_t batch = input.size(0);
        int64_t channels = input.size(1);
        int64_t timesteps = input.size(2);
    
        int64_t order = a.size(2)-1;

        resize(batch, channels, order);

        auto input_a = input.accessor<float, 3>();   // size (batch, time, channels)
        auto output_a = output.accessor<float, 3>(); // size (batch, time, channels)
        auto a_a = a.accessor<float, 3>(); // size (time, channels, order+1)
        auto b_a = b.accessor<float, 3>(); // size (time, channels, order+1)
        auto state_a = state.accessor<float, 3>(); // size (batch, channels, order)

        for (int64_t b = 0; b < batch; b++)
        {
            for (int64_t c = 0; c < channels; c++)
            {
                for (int64_t t = 0; t < timesteps; t++)
                {
                    float y = 0;
                    for (int64_t i = 0; i < channels; i++)
                    {
                        y += b_a[c][i] * input_a[b][t][i];
                        y += a_a[c][i] * state_a[b][t][i];
                    }
                    output_a[b][t][c] = y;
                    // update state
                    for (int64_t i = order-1; i > 0; i--)
                    {
                        state_a[b][c][i] = state_a[b][c][i-1];
                    }
                    state_a[b][c][0] = y;
                }
            }
        }
    }



private:

    void resize(size64_t batch, size64_t channels, size64_t order)
    {
        if (batch != state.size(0) || channels != state.size(1) || order != state.size(2))
        {
            state = torch::zeros({batch, channels, order});
        }
    }

    torch::Tensor state; // size (batch, channels, order)


};



} // glotnet

init_lfilter(py::module &m)
{
    py::class_<LFilter>(m, "LFilter")
        .def(py::init<>())
        .def("set_weight", &LFilter::setWeight)
        .def("set_bias", &LFilter::setBias)
        .def("forward", &LFilter::forward)
        .def("forward_cond", &LFilter::forward_cond);
}