import torch

from glotnet.convolution import Convolution

def test_causal_conv():
    print("Testing causal conv")
    torch.manual_seed(42)
    timesteps = 200
    impulse_loc = 50
    x = torch.zeros(1, 1, timesteps)
    x[..., impulse_loc] = 1.0
    conv = Convolution(in_channels=1, out_channels=1, kernel_size=50, dilation=1)
    y = conv(x)
    # causality check
    assert torch.allclose(y[..., :impulse_loc], conv.bias)
    print("   ok!")   

def test_forward_siso():
    print("Testing forward pass with Single-In-Single-Out, batch size 1")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    in_channels = 1
    out_channels= 1
    kernel_size = 50
    dilation = 1
    x = torch.randn(batch, in_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation)
    y1 = conv(x, training=True)
    y2 = conv(x, training=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5)
    print("   ok!")  

def test_forward_simo():
    print("Testing forward pass with Single-In-Multi-Out, batch size 1")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 1
    out_channels= 8
    kernel_size = 30
    dilation = 1
    x = torch.randn(batch, in_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation)
    y1 = conv(x, training=True)
    y2 = conv(x, training=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5)
    print("   ok!")  

def test_forward_miso():
    print("Testing forward pass with Multi-In-Single-Out, batch size 1")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 20
    out_channels= 1
    kernel_size = 30
    dilation = 1
    x = torch.randn(batch, in_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation)
    y1 = conv(x, training=True)
    y2 = conv(x, training=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5)
    print("   ok!")

def test_forward_mimo():
    print("Testing forward pass with Multi-In-Multi-Out, batch size 1")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 20
    out_channels= 50
    kernel_size = 30
    dilation = 1
    x = torch.randn(batch, in_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation)
    y1 = conv(x, training=True)
    y2 = conv(x, training=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5)
    print("   ok!")

def test_forward_mimo_dilated():
    print("Testing Dilated forward pass with Multi-In-Multi-Out, batch size 1")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 20
    out_channels= 50
    kernel_size = 30
    dilation = 4
    x = torch.randn(batch, in_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation)
    y1 = conv(x, training=True)
    y2 = conv(x, training=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5)
    print("   ok!")  

def test_forward_mimo_dilated_multibatch():
    print("Testing Dilated forward pass with Multi-In-Multi-Out, batch size 8")
    torch.manual_seed(42)
    timesteps = 100
    batch = 8
    in_channels = 20
    out_channels= 50
    kernel_size = 30
    dilation = 4
    x = torch.randn(batch, in_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation)
    y1 = conv(x, training=True)
    y2 = conv(x, training=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5)
    print("   ok!")  


if __name__ == "__main__":

    test_causal_conv()  
    test_forward_siso()
    test_forward_simo()
    test_forward_miso()
    test_forward_mimo()
    test_forward_mimo_dilated()
    test_forward_mimo_dilated_multibatch()




