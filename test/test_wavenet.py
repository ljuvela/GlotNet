import torch
from glotnet.wavenet import WaveNet

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


if __name__ == "__main__":
    test_wavenet_minimal()
    test_wavenet_cond_minimal()
#     test_stack_multichan()
#     test_stack_multilayer()
#     test_stack_multibatch()





