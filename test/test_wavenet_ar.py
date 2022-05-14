from typing import Sequence
from unittest import skip
import torch

from glotnet.wavenet_ar import WaveNetAR


def test_wavenet_ar_minimal():
    print("Test wavenet with minimal configuration")
    torch.manual_seed(42)
    timesteps = 2
    channels = 1
    residual_channels = 4
    skip_channels = 4

    kernel_size = 3
    dilations = [1]

    wavenet = WaveNetAR(input_channels=channels, output_channels=channels,
                        residual_channels=residual_channels, skip_channels=skip_channels,
                        kernel_size=kernel_size, dilations=dilations)
    y1 = wavenet(timesteps=timesteps, use_cpu=False)
    y2 = wavenet(timesteps=timesteps, use_cpu=True)

    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ref: {y1} \n ext: {y2}"
    print("   ok!")


def test_wavenet_ar_cond_minimal():
    print("Test wavenet with minimal configuration")
    torch.manual_seed(42)
    batch = 1
    timesteps = 2
    channels = 1
    residual_channels = 4
    skip_channels = 4
    cond_channels = 1

    kernel_size = 3
    dilations = [1]

    c = 0.1 * torch.randn(batch, cond_channels, timesteps)


    wavenet = WaveNetAR(input_channels=channels, output_channels=channels,
                        residual_channels=residual_channels, skip_channels=skip_channels,
                        kernel_size=kernel_size, dilations=dilations,
                        cond_channels=cond_channels)
    y1 = wavenet(cond_input=c, use_cpu=False)
    y2 = wavenet(cond_input=c, use_cpu=True)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ref: {y1} \n ext: {y2}"
    print("   ok!")


def test_wavenet_ar():
    print("Test wavenet with representative configuration")
    torch.manual_seed(42)
    timesteps = 100
    channels = 1
    residual_channels = 32
    skip_channels = 32

    kernel_size = 3
    dilations = [1, 2, 4, 8]

    wavenet = WaveNetAR(input_channels=channels, output_channels=channels,
                        residual_channels=residual_channels, skip_channels=skip_channels,
                        kernel_size=kernel_size, dilations=dilations)
    y1 = wavenet(timesteps=timesteps, use_cpu=False)
    y2 = wavenet(timesteps=timesteps, use_cpu=True)
    # assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
    #     f"Outputs must match \n ref: {y1} \n ext: {y2}"
    print("   ok!")



if __name__ == "__main__":
    test_wavenet_ar_minimal()
    test_wavenet_ar_cond_minimal()
    test_wavenet_ar()





