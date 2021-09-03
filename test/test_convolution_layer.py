import torch

from glotnet.convolution_layer import ConvolutionLayer

if __name__ == "__main__":
    print("Test layer with linear activations")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 1
    out_channels= 1
    kernel_size = 20
    dilation = 1
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels, kernel_size,
                            dilation=dilation, activation="linear")
    y1, s1 = conv(x, training=True)
    y2, s2 = conv(x, training=False)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")

# def test_forward1():
if __name__ == "__main__":
    print("Test layer with linear activations")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 1
    out_channels= 1
    kernel_size = 50
    dilation = 1
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels,
                            kernel_size, dilation=dilation, activation="tanh")
    y1, s1 = conv(x, training=True)
    y2, s2 = conv(x, training=False)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")

if __name__ == "__main__":
    print("Test layer with gated activations")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 1
    out_channels= 1
    kernel_size = 50
    dilation = 1
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels,
                            kernel_size, dilation=dilation, activation="gated")
    y1, s1 = conv(x, training=True)
    y2, s2 = conv(x, training=False)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"
    print("   ok!")

# if __name__ == "__main__":
    # print("Testing forward pass with SISO, batch size 1")
    # test_forward1()
    # print("   ok!")




