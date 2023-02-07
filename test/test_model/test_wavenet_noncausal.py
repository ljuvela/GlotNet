from glotnet.model.feedforward.wavenet import WaveNet
import torch

def test_wavenet_non_causal():
# if __name__ == "__main__":

    model = WaveNet(1, 1, residual_channels=8, skip_channels=8, kernel_size=3,
    dilations=[1, 2, 4, 8, 16, 32], causal=False)


    x = torch.zeros(1, 1, 2048)
    x[..., 1000] = 1.0

    y = model(x)

    receptive_field = model.receptive_field

    # from matplotlib import pyplot as plt

    # plt.plot(y.squeeze().detach())
    # plt.show()


def test_wavenet_multichannel():

    input_dim = 5
    output_dim = 10
    batch = 4
    frames = 50

    lar_target = torch.randn(batch, output_dim, frames)
    input = torch.randn(batch, input_dim, frames)

    model = WaveNet(
        input_channels=input_dim,
        output_channels=output_dim,
        residual_channels=8,
        skip_channels=8,
        kernel_size=3,
        dilations=[1, 2, 4, 8, 16, 32],
        causal=False)

    lar = model(input)

    assert lar.shape == lar_target.shape, \
        f"Expected shape {lar_target.shape}, got {lar.shape}"
    

if __name__ == "__main__":
    test_wavenet_multichannel()