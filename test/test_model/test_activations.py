import torch
import glotnet
from glotnet.model.feedforward.activations import Activation, ActivationType

def test_linear():
    torch.manual_seed(42)
    timesteps = 10
    batch = 2
    in_channels = 1

    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    act = Activation(activation=ActivationType.linear)
    y1 = act(x, use_extension=True)
    y2 = act(x, use_extension=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), f"Outputs must match\n  y1={y1},  y2={y2}"


def test_gated():
    torch.manual_seed(42)
    timesteps = 10
    batch = 4
    in_channels = 16

    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    act = Activation(activation=ActivationType.gated)
    y1 = act(x, use_extension=True)
    y2 = act(x, use_extension=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), f"Outputs must match\n  y1={y1},  y2={y2}"


def test_custom_tanh():

    tanh =  glotnet.model.feedforward.activations.Tanh()
    tanh_ref = torch.nn.Tanh()

    # test forward pass
    x = torch.randn(10, 10, 10).requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)
    y = tanh(x)
    y_ref = tanh_ref(x_ref)
    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5), \
        f"Outputs must match\n  y={y},  y_ref={y_ref}"

    # test backward pass
    y.sum().backward()
    y_ref.sum().backward()
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-5), \
        f"Gradients must match\n  x.grad={x.grad},  x.grad={x_ref.grad}"