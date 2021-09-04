import torch

from glotnet.convolution_stack import ConvolutionStack

if __name__ == "__main__":
# def test_stack_minimal():
    print("Test stack with minimal configuration")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    channels = 1
    kernel_size = 20
    dilations = [1,]
    x = 0.1 * torch.randn(batch, channels, timesteps)

    stack = ConvolutionStack(channels, kernel_size, dilations=[1],
                 activation="gated",use_residual=True)
    y1, s1 = stack(x, training=True)
    y2, s2 = stack(x, training=False)
    assert torch.allclose(s1[0], s2[0], atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")





# if __name__ == "__main__":
    # test_stack_minimal()




