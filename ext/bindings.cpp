#include <torch/extension.h>

void init_convolution(py::module&);
void init_convolution_layer(py::module&);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  init_convolution(m);
  init_convolution_layer(m);
}
