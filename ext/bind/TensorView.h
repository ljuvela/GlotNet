#pragma once

#include <torch/extension.h>

#include "glotnet/ArrayView.h"

namespace glotnet
{

/// Holds a contiguous float32 CPU tensor and converts to ConstArrayView.
///
/// Intended to be used as a temporary in a call to the core library, e.g.
///   conv.setWeight(glotnet::view(weight));
/// The temporary (and so the tensor data) lives until the call returns.
class TensorView
{
public:
    explicit TensorView(const torch::Tensor &t)
    {
        TORCH_CHECK(t.scalar_type() == torch::kFloat32,
                    "expected a float32 tensor, got ", t.scalar_type());
        TORCH_CHECK(t.device().is_cpu(),
                    "expected a CPU tensor, got ", t.device());
        tensor = t.detach().contiguous();
    }

    operator ConstArrayView() const
    {
        return ConstArrayView(tensor.data_ptr<float>(), tensor.sizes().vec());
    }

private:
    torch::Tensor tensor;
};

inline TensorView view(const torch::Tensor &t) { return TensorView(t); }

} // namespace glotnet
