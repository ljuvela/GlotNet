#include <torch/extension.h>
#include <vector>

#include "glotnet/Distributions.h"

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
    auto temp_tensor = torch::ones({batch_size, timesteps, 1u}) * temperature;
    auto temp_a = temp_tensor.accessor<float, 3>();
    for (int64_t b = 0; b < batch_size; b++)
    {
        dist.sample(&params_a[b][0][0],
                    &x_a[b][0][0],
                    timesteps, 
                    &temp_a[b][0][0]);
    }

    return {x};
}

std::vector<at::Tensor> sample_categorical(
    const torch::Tensor &params, const torch::Tensor &temperature,
    int64_t num_bits, int64_t seed)
{
    // params: logits (batch, time, 2^num_bits); temperature: (batch, time, 1)
    // seed < 0: nondeterministic
    TORCH_CHECK(params.dim() == 3 && params.scalar_type() == torch::kFloat32
                && params.is_contiguous() && params.device().is_cpu(),
                "params must be a contiguous float32 CPU tensor (batch, time, classes)");
    const int64_t batch_size = params.size(0);
    const int64_t timesteps = params.size(1);
    TORCH_CHECK(temperature.scalar_type() == torch::kFloat32 && temperature.is_contiguous()
                && temperature.sizes() == torch::IntArrayRef({batch_size, timesteps, 1}),
                "temperature must be a contiguous float32 tensor (batch, time, 1)");

    auto dist = CategoricalDensity(static_cast<int>(num_bits));
    TORCH_CHECK(params.size(2) == static_cast<int64_t>(dist.paramsDim()),
                "expected ", dist.paramsDim(), " classes, got ", params.size(2));
    if (seed >= 0)
        dist.seed(static_cast<uint64_t>(seed));

    auto x = torch::zeros({batch_size, timesteps, 1});
    const float *params_data = params.data_ptr<float>();
    const float *temp_data = temperature.data_ptr<float>();
    float *x_data = x.data_ptr<float>();
    const int64_t classes = params.size(2);
    for (int64_t b = 0; b < batch_size; b++)
    {
        dist.sample(params_data + b * timesteps * classes,
                    x_data + b * timesteps,
                    timesteps,
                    temp_data + b * timesteps);
    }
    return {x};
}

} // distributions
} // glotnet

void init_distributions(py::module &m)
{
    m.def("sample_gaussian", &(glotnet::distributions::sample_gaussian), "Sample from Gaussian distribution");
    m.def("sample_categorical", &(glotnet::distributions::sample_categorical),
          "Sample from mu-law categorical distribution",
          py::arg("params"), py::arg("temperature"), py::arg("num_bits") = 8, py::arg("seed") = -1);
}