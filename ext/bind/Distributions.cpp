#include <torch/extension.h>
#include <vector>

#include "../src/Distributions.h"

namespace glotnet
{
namespace distributions
{

std::vector<at::Tensor> sample_gaussian(
    const torch::Tensor & params, float temperature)
{
    int64_t batch_size = params.size(0);
    int64_t timesteps = params.size(1);
    int64_t channels = params.size(2);

    assert (channels == 2);

    auto x = torch::zeros({batch_size, timesteps, 1u});

    auto params_a  =  params.accessor<float, 3>();  // size (batch, time, 2)
    auto x_a = x.accessor<float, 3>(); // size (batch, time, 1)

    auto dist = GaussianDensity(-7.0);
    float temp = 1.0f;
    for (int64_t b = 0; b < batch_size; b++)
    {
        dist.sample(&params_a[b][0][0],
                    &x_a[b][0][0],
                    timesteps, &temp);
    }

    return {x};
}

} // distributions
} // glotnet

void init_distributions(py::module &m)
{
    m.def("sample_gaussian", &(glotnet::distributions::sample_gaussian), "Sample from Gaussian distribution");
}