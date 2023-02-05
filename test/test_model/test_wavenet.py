import torch
from glotnet.model.feedforward.wavenet import WaveNet

def test_wavenet_minimal():
    print("Test wavenet with minimal configuration")
    torch.manual_seed(42)
    timesteps = 10
    batch = 1
    input_channels = 1
    output_channels = 1
    residual_channels = 4
    skip_channels = 3

    kernel_size = 3
    dilations = [1]
    x = 0.1 * torch.randn(batch, input_channels, timesteps)

    wavenet = WaveNet(input_channels=input_channels, output_channels=output_channels,
                      residual_channels=residual_channels, skip_channels=skip_channels,
                      kernel_size=kernel_size, dilations=dilations)
    y1 = wavenet(x, sequential=False)
    y2 = wavenet(x, sequential=True)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ref: {y1} \n ext: {y2}"
    print("   ok!")


def test_wavenet_cond_minimal():
    print("Test conditional wavenet with minimal configuration")
    torch.manual_seed(42)
    timesteps = 10
    batch = 1
    input_channels = 1
    output_channels = 1
    residual_channels = 4
    cond_channels = 1
    skip_channels = 3
    kernel_size = 3
    dilations = [1]
    x = 0.1 * torch.randn(batch, input_channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    wavenet = WaveNet(input_channels=input_channels, output_channels=output_channels,
                      residual_channels=residual_channels, skip_channels=skip_channels,
                      kernel_size=kernel_size, dilations=dilations,
                      cond_channels=cond_channels)
    y1 = wavenet(x, cond_input=c, sequential=False)
    y2 = wavenet(x, cond_input=c, sequential=True)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ref: {y1} \n ext: {y2}"
    print("   ok!")


def test_wavenet_multichan():
    print("Test wavenet with multi-channel configuration")
    torch.manual_seed(42)
    timesteps = 10
    batch = 1
    input_channels = 2
    output_channels = 4
    residual_channels = 5
    skip_channels = 6
    kernel_size = 3
    dilations = [1]
    x = 0.1 * torch.randn(batch, input_channels, timesteps)

    wavenet = WaveNet(input_channels=input_channels, output_channels=output_channels,
                      residual_channels=residual_channels, skip_channels=skip_channels,
                      kernel_size=kernel_size, dilations=dilations)
    y1 = wavenet(x, sequential=True)
    y2 = wavenet(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Main outputs must match \n ext: {y1} \n ref {y2}"
    print("   ok!")



def test_wavenet_multilayer():
    print("Test wavenet with multi-layer configuration")
    torch.manual_seed(42)
    timesteps = 10
    batch = 1
    input_channels = 2
    output_channels = 4
    residual_channels = 5
    skip_channels = 6
    kernel_size = 3
    dilations = [1, 2, 3, 4, 8]
    x = 0.1 * torch.randn(batch, input_channels, timesteps)

    wavenet = WaveNet(input_channels=input_channels, output_channels=output_channels,
                      residual_channels=residual_channels, skip_channels=skip_channels,
                      kernel_size=kernel_size, dilations=dilations)
    y1 = wavenet(x, sequential=True)
    y2 = wavenet(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Main outputs must match \n ext: {y1} \n ref {y2}"
    print("   ok!")


def test_wavenet_multibatch():
    print("Test wavenet with multi-batch configuration")
    torch.manual_seed(42)
    timesteps = 10
    batch = 18
    input_channels = 2
    output_channels = 4
    residual_channels = 5
    skip_channels = 6
    kernel_size = 3
    dilations = [1, 2, 3, 4, 8]
    x = 0.1 * torch.randn(batch, input_channels, timesteps)

    wavenet = WaveNet(input_channels=input_channels, output_channels=output_channels,
                      residual_channels=residual_channels, skip_channels=skip_channels,
                      kernel_size=kernel_size, dilations=dilations)
    y1 = wavenet(x, sequential=True)
    y2 = wavenet(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Main outputs must match \n ext: {y1} \n ref {y2}"
    print("   ok!")


def test_wavenet_cond_multibatch():
    print("Test conditional wavenet with multi-batch configuration")
    torch.manual_seed(42)
    timesteps = 101
    batch = 17
    input_channels = 3
    output_channels = 5
    residual_channels = 16
    cond_channels = 52
    skip_channels = 12
    kernel_size = 3
    dilations = [1, 2, 3, 17]
    x = 0.1 * torch.randn(batch, input_channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    wavenet = WaveNet(input_channels=input_channels, output_channels=output_channels,
                      residual_channels=residual_channels, skip_channels=skip_channels,
                      kernel_size=kernel_size, dilations=dilations,
                      cond_channels=cond_channels)
    y1 = wavenet(x, cond_input=c, sequential=False)
    y2 = wavenet(x, cond_input=c, sequential=True)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ref: {y1} \n ext: {y2}"
    print("   ok!")





