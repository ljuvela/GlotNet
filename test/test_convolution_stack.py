import torch

from glotnet.convolution_stack import ConvolutionStack

def test_stack_minimal():
    print("Test stack with minimal configuration")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    channels = 1
    kernel_size = 20
    dilations = [1,]
    x = 0.1 * torch.randn(batch, channels, timesteps)

    stack = ConvolutionStack(channels, kernel_size, dilations=dilations,
                 activation="gated",use_residual=True)
    y1, s1 = stack(x, sequential=True)
    y2, s2 = stack(x, sequential=False)
    assert torch.allclose(s1[0], s2[0], atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")

def test_stack_cond_minimal():
    print("Test conditional stack with minimal configuration")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    channels = 1
    cond_channels = 1
    kernel_size = 20
    dilations = [1,]
    x = 0.1 * torch.randn(batch, channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    stack = ConvolutionStack(channels, kernel_size, dilations=dilations,
                 activation="gated", use_residual=True, cond_channels=cond_channels)
    y1, s1 = stack(x, cond_input=c, sequential=True)
    y2, s2 = stack(x, cond_input=c, sequential=False)
    assert torch.allclose(s1[0], s2[0], atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")


def test_stack_multichan():
    print("Test stack with multi-channel configuration")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    channels = 32
    kernel_size = 2
    dilations = [1,]
    x = 0.1 * torch.randn(batch, channels, timesteps)

    stack = ConvolutionStack(channels, kernel_size, dilations=dilations,
                 activation="gated",use_residual=True)
    y1, s1 = stack(x, sequential=True)
    y2, s2 = stack(x, sequential=False)
    assert torch.allclose(s1[0], s2[0], atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")



def test_stack_multilayer():
    print("Test stack with multi-layer configuration")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    channels = 32
    kernel_size = 2
    dilations = [1, 2, 4, 8, 16]
    x = 0.1 * torch.randn(batch, channels, timesteps)

    stack = ConvolutionStack(channels, kernel_size, dilations=dilations,
                 activation="gated", use_residual=True)
    y1, skips1 = stack(x, sequential=True)
    y2, skips2 = stack(x, sequential=False)
    for s1, s2 in zip(skips1, skips2):
        assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")


def test_stack_cond_multilayer():
    print("Test stack with conditional multi-layer configuration")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    channels = 32
    cond_channels = 7
    kernel_size = 2
    dilations = [1, 2, 4, 8, 16]
    x = 0.1 * torch.randn(batch, channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    stack = ConvolutionStack(channels, kernel_size, dilations=dilations,
                 activation="gated", use_residual=True,
                 cond_channels=cond_channels)

    y1, skips1 = stack(x, cond_input=c, sequential=True)
    y2, skips2 = stack(x, cond_input=c, sequential=False)
    for i, (s1, s2) in enumerate(zip(skips1, skips2)):
        assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"
        print(f"skip layer {i} ok")
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")


def test_stack_multibatch():
    print("Test stack with larger batch configuration")
    torch.manual_seed(42)
    timesteps = 100
    batch = 8
    channels = 32
    kernel_size = 4
    dilations = [1, 2, 4, 8, 16]
    x = 0.1 * torch.randn(batch, channels, timesteps)

    stack = ConvolutionStack(channels, kernel_size, dilations=dilations,
                 activation="gated",use_residual=True)
    y1, s1 = stack(x, sequential=True)
    y2, s2 = stack(x, sequential=False)
    assert torch.allclose(s1[0], s2[0], atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")



if __name__ == "__main__":
    test_stack_minimal()
    test_stack_cond_minimal()
    test_stack_multichan()
    test_stack_multilayer()
    test_stack_cond_multilayer()
    test_stack_multibatch()




