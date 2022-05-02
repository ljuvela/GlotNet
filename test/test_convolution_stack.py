from unittest import skip
import torch

from glotnet.convolution_stack import ConvolutionStack

def test_stack_minimal():
    print("Test stack with minimal configuration")
    torch.manual_seed(42)
    timesteps = 1
    batch = 1
    channels = 1
    skip_channels = 2
    kernel_size = 20
    dilations = [1,]
    x = 0.1 * torch.randn(batch, channels, timesteps)
    stack = ConvolutionStack(channels=channels, skip_channels=skip_channels,
                             kernel_size=kernel_size,
                             dilations=dilations,
                             activation="gated",
                             use_residual=True)
    y1, s1 = stack(x, sequential=True)
    y2, s2 = stack(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Main outputs must match \n ext: {y1} \n ref {y2}"
    assert torch.allclose(s1[0], s2[0], atol=1e-6, rtol=1e-5), \
        f"Skip outputs must match \n ext: {s1}, \n ref {s2}"
    print("   ok!")

def test_stack_cond_minimal():
    print("Test conditional stack with minimal configuration")
    torch.manual_seed(42)
    timesteps = 2
    batch = 1
    channels = 1
    skip_channels = 2
    cond_channels = 1
    kernel_size = 20
    dilations = [1,]
    x = 0.1 * torch.randn(batch, channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    stack = ConvolutionStack(channels=channels, skip_channels=skip_channels,
                             kernel_size=kernel_size, dilations=dilations,
                             activation="gated", use_residual=True, cond_channels=cond_channels)
    y1, s1 = stack(x, cond_input=c, sequential=True)
    y2, s2 = stack(x, cond_input=c, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Main outputs must match \n ext: {y1} \n ref {y2}"
    assert torch.allclose(s1[0], s2[0], atol=1e-6, rtol=1e-5), \
        f"Skip outputs must match \n ext: {s1} \n ref {s2}"
    print("   ok!")


def test_stack_multichan():
    print("Test stack with multi-channel configuration")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    channels = 32
    skip_channels = 16
    kernel_size = 2
    dilations = [1,]
    x = 0.1 * torch.randn(batch, channels, timesteps)

    stack = ConvolutionStack(channels=channels, skip_channels=skip_channels,
                             kernel_size=kernel_size, dilations=dilations,
                             activation="gated", use_residual=True)
    y1, s1 = stack(x, sequential=True)
    y2, s2 = stack(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Main outputs must match \n ext: {y1} \n ref {y2}"
    assert torch.allclose(s1[0], s2[0], atol=1e-6, rtol=1e-5), \
        f"Skip outputs must match \n ext: {s1} \n ref {s2}"
    print("   ok!")


def test_stack_multilayer():
    print("Test stack with multi-layer configuration")
    torch.manual_seed(42)
    timesteps = 5
    batch = 1
    channels = 2
    skip_channels = 4
    kernel_size = 2
    # dilations = [1, 2, 4, 8, 16]
    dilations = [1, 2]
    x = 0.1 * torch.randn(batch, channels, timesteps)

    stack = ConvolutionStack(channels=channels, skip_channels=skip_channels,
                             kernel_size=kernel_size, dilations=dilations,
                             activation="gated", use_residual=True)
    y1, skips1 = stack(x, sequential=True)
    y2, skips2 = stack(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Main outputs should match"
    for i, (s1, s2) in enumerate(zip(skips1, skips2)):
        assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), \
            f"Skip outputs should match. \n Got sizes {s1.shape} and {s2.shape}. \n Got values\n ext: {s1}\n ref: {s2}" 
    print("   ok!")


def test_stack_cond_multilayer():
    print("Test stack with conditional multi-layer configuration")
    torch.manual_seed(42)
    timesteps = 3
    batch = 1
    channels = 1
    skip_channels = 1
    cond_channels = 1
    kernel_size = 2
    # dilations = [1, 2, 4, 8, 16]
    dilations = [1, 2]
    x = 0.1 * torch.randn(batch, channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    stack = ConvolutionStack(channels=channels, skip_channels=skip_channels,
                             kernel_size=kernel_size, dilations=dilations,
                             activation="gated", use_residual=True,
                             cond_channels=cond_channels)

    y1, skips1 = stack(x, cond_input=c, sequential=True)
    y2, skips2 = stack(x, cond_input=c, sequential=False)
    for i, (s1, s2) in enumerate(zip(skips1, skips2)):
        assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), \
        f"Skip outputs ({i}) must match \n ext: {s1} \n ref: {s2}"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Main outputs must match \n ext: {y1} \n ref: {y2}"

    print("   ok!")


def test_stack_multibatch():
    print("Test stack with larger batch configuration")
    torch.manual_seed(42)
    timesteps = 3
    batch = 2
    channels = 4
    skip_channels = 5
    kernel_size = 2
    dilations = [1, 2, 4, 8, 16]
    x = 0.1 * torch.randn(batch, channels, timesteps)

    stack = ConvolutionStack(channels=channels, skip_channels=skip_channels,
                             kernel_size=kernel_size, dilations=dilations,
                             activation="gated", use_residual=True)
    y1, skips1 = stack(x, sequential=True)
    y2, skips2 = stack(x, sequential=False)
    for i, (s1, s2) in enumerate(zip(skips1, skips2)):
        assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), \
        f"Skip outputs (layer {i}) must match \n ext: {s1} \n ref: {s2}"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Main outputs must match \n ext: {y1} \n ref: {y2}"

    print("   ok!")



if __name__ == "__main__":
    test_stack_minimal()
    test_stack_cond_minimal()
    test_stack_multichan()
    test_stack_multilayer()
    test_stack_cond_multilayer()
    test_stack_multibatch()





