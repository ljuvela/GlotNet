import torch

from glotnet.model.wavenet import WaveNet
from glotnet.model.wavenet_ar import WaveNetAR

if __name__ == "__main__":
    print("Test parameter migration between AR and non-AR wavenets")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    input_channels = 1
    output_channels = 1
    residual_channels = 4

    kernel_size = 3
    dilations = [1, 2, 4]
    x = 0.1 * torch.randn(batch, input_channels, timesteps)

    wavenet = WaveNet(input_channels, output_channels,
                        residual_channels, kernel_size,
                        dilations=dilations)

    wavenet_ar = WaveNetAR(input_channels, output_channels,
                        residual_channels, kernel_size,
                        dilations=dilations)

    wavenet_ar.load_state_dict(wavenet.state_dict())
