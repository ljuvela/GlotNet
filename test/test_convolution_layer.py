import torch

from glotnet.convolution_layer import ConvolutionLayer

def test_layer_linear():
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
    y1, s1 = conv(x, sequential=True)
    y2, s2 = conv(x, sequential=False)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"

def test_layer_tanh():
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
    y1, s1 = conv(x, sequential=True)
    y2, s2 = conv(x, sequential=False)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"


def test_layer_gated_no_out_transform():
    
    torch.manual_seed(42)
    timesteps = 50
    batch = 5
    in_channels = 4
    out_channels= 2
    kernel_size = 3
    dilation = 1
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels,
                            kernel_size, dilation=dilation, 
                            activation="gated", use_output_transform=False)
    
    y1, s1 = conv(x, sequential=True)
    y2, s2 = conv(x, sequential=False)

    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"

def test_layer_gated():
    torch.manual_seed(42)
    timesteps = 30
    batch = 4
    in_channels = 4
    out_channels= 3
    kernel_size = 5
    dilation = 4
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels,
                            kernel_size, dilation=dilation, activation="gated")
    y1, s1 = conv(x, sequential=True)
    y2, s2 = conv(x, sequential=False)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), f"Skip outputs must match:\n s1={s1}\n s2={s2}"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), f"Main outputs must match:\n y1={y1}\n y2={y2}"

def test_layer_gated_cond():
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 1
    out_channels= 1
    cond_channels = 3
    kernel_size = 50
    dilation = 1
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels, kernel_size,
                            dilation=dilation, activation="gated", cond_channels=cond_channels)
    y1, s1 = conv(x, cond_input=c, sequential=False)
    y2, s2 = conv(x, cond_input=c, sequential=True)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"

if __name__ == "__main__":

    print("Test layer with linear activations")
    test_layer_linear()
    print("   ok!")

    print("Test layer with tanh activation")
    test_layer_tanh()
    print("   ok!")

    print("Test layer with gated activations, no output transform")
    test_layer_gated_no_out_transform()
    print("   ok!")

    print("Test layer with gated activations")
    test_layer_gated()
    print("   ok!")

    print("Test conditional layer with gated activations")
    test_layer_gated_cond()
    print("   ok!")





