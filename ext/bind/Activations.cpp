//    Copyright 2022 Lauri Juvela

//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at

//        http://www.apache.org/licenses/LICENSE-2.0

//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.

#include <torch/extension.h>
#include <vector>
#include <iostream>

#include "../src/Activations.h"

namespace glotnet
{
namespace activations
{

using Activations::getActivationFuncArray;
using namespace torch::indexing;

std::vector<at::Tensor> forward(torch::Tensor & input, std::string & activation_type)
{
    int64_t batch_size = input.size(0);
    int64_t timesteps = input.size(1);
    int64_t channels = input.size(2);

    // std::cout << "Channels " << channels << std::endl;

    const bool is_gated = Activations::isGated(activation_type);

    auto output = input * 1.0f; // copy

    float * data = output.data_ptr<float>();

    auto act = getActivationFuncArray(activation_type);

    for (int64_t b = 0; b < batch_size; b++)
    {
        act(&(data[b * channels * timesteps]), timesteps, channels);
    }

    if (is_gated)
        return {output.index({Slice(), Slice(), Slice(None, channels / 2)})};
    else
        return {output};
}

} // EOF activations
} // EOF glotnet

void init_activations(py::module &m)
{
    m.def("activations_forward", &(glotnet::activations::forward), "Activations forward");
}
