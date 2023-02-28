import torch

from glotnet.model.autoregressive.wavenet import WaveNetAR
from glotnet.losses.distributions import GaussianDensity

def test_wavenet_ar_minimal():
    print("Test AR wavenet with minimal configuration")
    torch.manual_seed(42)
    batch = 1
    timesteps = 10
    channels = 1
    residual_channels = 4
    skip_channels = 4

    kernel_size = 3
    dilations = [1]

    dist = GaussianDensity()
    wavenet = WaveNetAR(input_channels=channels, output_channels=2 * channels,
                        residual_channels=residual_channels, skip_channels=skip_channels,
                        kernel_size=kernel_size, dilations=dilations,
                        distribution=dist)

    x = torch.zeros(batch, channels, timesteps)
    temp = torch.zeros(batch, 1, timesteps)
    y_ref = wavenet.forward(input=x, temperature=temp)
    y_ext = wavenet.inference(input=x, temperature=temp)

    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"
    print("   ok!")


def test_wavenet_ar_cond_minimal():
    print("Test conditional AR wavenet with minimal configuration")
    torch.manual_seed(42)
    batch = 1
    timesteps = 3
    channels = 1
    residual_channels = 4
    skip_channels = 4
    cond_channels = 1

    kernel_size = 3
    dilations = [1]

    dist = GaussianDensity(temperature=0.0)
    wavenet = WaveNetAR(input_channels=channels, output_channels=2*channels,
                        residual_channels=residual_channels, skip_channels=skip_channels,
                        kernel_size=kernel_size, dilations=dilations,
                        cond_channels=cond_channels,
                        distribution=dist)

    x = torch.zeros(batch, channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    temp = torch.zeros(batch, 1, timesteps)
    y_ref = wavenet.forward(input=x, cond_input=c, temperature=temp)
    y_ext = wavenet.inference(input=x, cond_input=c, temperature=temp)

    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"
    print("   ok!")


def test_wavenet_ar():
    print("Test wavenet AR with representative configuration")
    torch.manual_seed(42)
    batch = 1
    timesteps = 50
    channels = 1
    residual_channels = 32
    skip_channels = 32

    kernel_size = 2
    dilations = [1, 2, 4]

    dist = GaussianDensity()
    wavenet = WaveNetAR(input_channels=channels, output_channels=2*channels,
                        residual_channels=residual_channels, skip_channels=skip_channels,
                        kernel_size=kernel_size, dilations=dilations, distribution=dist)
    receptive_field = wavenet.receptive_field

    x = torch.zeros(batch, channels, timesteps)
    temp = torch.zeros(batch, 1, timesteps)
    y_ref = wavenet.forward(input=x, temperature=temp)
    y_ext = wavenet.inference(input=x, temperature=temp)

    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"
    print("   ok!")


def test_wavenet_ar_cond():
    print("Test conditional AR wavenet with representative configuration")
    torch.manual_seed(42)
    batch = 1
    timesteps = 50
    channels = 1
    residual_channels = 32
    skip_channels = 32
    cond_channels = 1

    kernel_size = 3
    dilations = [1]

    dist = GaussianDensity()
    wavenet = WaveNetAR(input_channels=channels, output_channels=2*channels,
                        residual_channels=residual_channels, skip_channels=skip_channels,
                        kernel_size=kernel_size, dilations=dilations,
                        cond_channels=cond_channels,
                        distribution=dist)
    receptive_field = wavenet.receptive_field

    x = torch.zeros(batch, channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    temp = torch.zeros(batch, 1, timesteps)
    y_ref = wavenet.forward(input=x, cond_input=c, temperature=temp)
    y_ext = wavenet.inference(input=x, cond_input=c, temperature=temp)

    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"
    print("   ok!")





