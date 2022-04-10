#include <torch/extension.h>

void init_convolution(py::module&);
void init_convolution_layer(py::module&);
void init_convolution_stack(py::module&);
void init_wavenet(py::module&);
void init_wavenet_ar(py::module&);


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  init_convolution(m);
  init_convolution_layer(m);
  init_convolution_stack(m);
  init_wavenet(m);
  init_wavenet_ar(m);
}
