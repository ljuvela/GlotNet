#include <torch/extension.h>

void init_convolution(py::module&);
void init_activations(py::module&);
void init_convolution_layer(py::module&);
void init_convolution_stack(py::module&);
void init_wavenet(py::module&);
void init_distributions(py::module&);
#include "bind/LFilter.h"
#include "bind/GlotNetAR.h"
#include "bind/WaveNetAR.h"

PYBIND11_MODULE(cpp_extensions, m)
{
    init_convolution(m);
    init_activations(m);
    init_convolution_layer(m);
    init_convolution_stack(m);
    init_wavenet(m);
    init_wavenet_ar(m);
    init_distributions(m);
    init_lfilter(m);
    init_glotnet_ar(m);
}
