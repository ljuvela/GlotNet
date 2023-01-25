import torch
from glotnet.model.feedforward.activations import Activation

def test_linear():
    print("Test linear activation")
    torch.manual_seed(42)
    timesteps = 10
    batch = 2
    in_channels = 1

    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    act = Activation(activation="linear")
    y1 = act(x, use_extension=True)
    y2 = act(x, use_extension=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), f"Outputs must match\n  y1={y1},  y2={y2}"
    print("   ok!")


def test_gated():
    print("Test linear activation")
    torch.manual_seed(42)
    timesteps = 10
    batch = 4
    in_channels = 16

    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    act = Activation(activation="gated")
    y1 = act(x, use_extension=True)
    y2 = act(x, use_extension=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), f"Outputs must match\n  y1={y1},  y2={y2}"
    print("   ok!")
