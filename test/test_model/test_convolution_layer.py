import torch
from glotnet.model.feedforward.convolution_layer import ConvolutionLayer

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
    y1 = conv(x, sequential=True)
    y2 = conv(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"

def test_layer_linear_skips():
    torch.manual_seed(42)
    timesteps = 10
    batch = 8
    in_channels = 2
    out_channels = 4
    skip_channels = 5
    kernel_size = 6
    dilation = 7
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels, kernel_size,
                            dilation=dilation, activation="linear", 
                            skip_channels=skip_channels)
    y1, s1 = conv(x, sequential=True)
    y2, s2 = conv(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Main outputs should match \n ext: {y1} \n ref: {y2}"
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), \
        f"Skip outputs should match \n ext: {s1} \n ref: {s2}"

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
    y1 = conv(x, sequential=True)
    y2 = conv(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Assert main outputs match"


def test_layer_gated_no_out_transform():
    
    torch.manual_seed(42)
    timesteps = 50
    batch = 5
    in_channels = 4
    out_channels= 2
    skip_channels = 6
    kernel_size = 3
    dilation = 1
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels,
                            kernel_size, dilation=dilation, 
                            activation="gated", use_output_transform=False,
                            skip_channels=skip_channels)
    
    y1, s1 = conv(x, sequential=True)
    y2, s2 = conv(x, sequential=False)

    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), "Main outputs should match"
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), "Assert skip output match"


def test_layer_gated():
    torch.manual_seed(42)
    timesteps = 30
    batch = 4
    in_channels = 4
    out_channels= 3
    kernel_size = 5
    skip_channels = 6
    dilation = 4
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels,
                            kernel_size, dilation=dilation, activation="gated",
                            skip_channels=skip_channels)
    y1, s1 = conv(x, sequential=True)
    y2, s2 = conv(x, sequential=False)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), f"Skip outputs must match:\n s1={s1}\n s2={s2}"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), f"Main outputs must match:\n y1={y1}\n y2={y2}"

def test_layer_gated_cond():
    torch.manual_seed(42)
    timesteps = 2
    batch = 1 
    in_channels = 1
    out_channels= 1
    cond_channels = 3
    skip_channels = 2
    kernel_size = 1
    dilation = 1
    x = 0.1 * torch.randn(batch, in_channels, timesteps)
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels, kernel_size,
                            dilation=dilation, activation="gated",
                            cond_channels=cond_channels, skip_channels=skip_channels)
    y1, s1 = conv(x, cond_input=c, sequential=False)
    y2, s2 = conv(x, cond_input=c, sequential=True)
    assert torch.allclose(s1, s2, atol=1e-6, rtol=1e-5), f"Skip outputs must match:\n ref={s1}\n ext={s2}"
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), f"Main outputs must match:\n ref={y1}\n ext={y2}"

if __name__ == "__main__":

    print("Test layer with linear activations")
    test_layer_linear()
    print("   ok!")

    print("Test layer with linear activations and skip transform")
    test_layer_linear_skips()
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





