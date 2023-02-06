import torch
from glotnet.model.feedforward.convolution import Convolution
from glotnet.model.autoregressive.convolution import ConvolutionAR

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

def empirical_receptive_field(model, timesteps):
    x = torch.zeros(1, 1, 2*timesteps, requires_grad=True)
    y = model(x)
    y[..., timesteps//2].backward()
    nonzero = (x.grad.abs() > 1e-9) * 1
    first = torch.argmax(nonzero)
    last = nonzero.size(-1) - torch.argmax(nonzero.flip(dims=(-1,)))
    return last-first

def test_receptive_field():
    print("Testing receptive field")
    torch.manual_seed(42)

    conv = Convolution(in_channels=1, out_channels=1, kernel_size=50, dilation=1)
    r = empirical_receptive_field(conv, timesteps=100)
    assert r == conv.receptive_field, \
        "Analytical and empiric receptive field lengths must match"

    conv = Convolution(in_channels=1, out_channels=1, kernel_size=3, dilation=4)
    r = empirical_receptive_field(conv, timesteps=100)
    assert r == conv.receptive_field, \
        f"Analytical and empiric receptive field lengths must match \n emp: {r} \n ana: {conv.receptive_field}"

    print("  ok!")


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
    y1 = conv(x, sequential=True)
    y2 = conv(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"
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
    y1 = conv(x, sequential=True)
    y2 = conv(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"
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
    y1 = conv(x, sequential=True)
    y2 = conv(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"
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
    y1 = conv(x, sequential=True)
    y2 = conv(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"
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
    y1 = conv(x, sequential=True)
    y2 = conv(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"
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
    y1 = conv(x, sequential=True)
    y2 = conv(x, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"
    print("   ok!")  


def test_forward_cond_siso():
    print("Testing conditional forward pass with Single-In-Single-Out, batch size 1")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1
    in_channels = 1
    out_channels= 1
    kernel_size = 50
    dilation = 1
    x = torch.randn(batch, in_channels, timesteps)
    c = torch.randn(batch, out_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation)
    y1 = conv(x, c, sequential=True)
    y2 = conv(x, c, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"
    print("   ok!")


def test_forward_cond_mimo():
    print("Testing conditional forward pass with Multi-In-Multi-Out, batch size 1")
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 20
    out_channels= 50
    kernel_size = 30
    dilation = 1
    x = torch.randn(batch, in_channels, timesteps)
    c = torch.randn(batch, out_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation)
    y1 = conv(x, c, sequential=True)
    y2 = conv(x, c, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"
    print("   ok!")

def test_conv_ar():

    print("Testing autoregressive conv for exponential decay")

    channels = 1
    timesteps = 10
    batch = 1

    x = torch.zeros(batch, channels, timesteps)
    x[..., 0] = 1.0

    layer = ConvolutionAR(channels, channels, 2, bias=False)

    # exponential decay
    coeff = 0.9
    kernel = torch.tensor([coeff])
    layer.weight.data = kernel.reshape(1, 1, -1)

    y = layer.forward(x)

    y_ref = coeff ** torch.arange(timesteps)

    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5)
    print("  ok!")


def test_conv_ar_extension():

    print("Testing autoregressive conv extension")

    channels = 1
    timesteps = 10
    batch = 1

    x = torch.zeros(batch, channels, timesteps)
    x[..., 0] = 1.0

    layer = ConvolutionAR(channels, channels, 1)

    # exponential decay
    coeff = 0.9
    kernel = torch.tensor([coeff])
    layer.weight.data = kernel.reshape(1, 1, -1)
    layer.bias.data.zero_()

    y_ref = layer.forward(x, use_extension=False)
    y = layer.forward(x, use_extension=True)

    assert torch.allclose(y, y_ref, atol=1e-6, rtol=1e-5), \
        f"Outputs must match, \n ref: {y_ref} \n ext: {y}"
    print("  ok!")

def test_forward_cond_film_siso():
    print("Testing film conditional forward pass with Single-In-Single-Out, batch size 1")
    torch.manual_seed(42)
    timesteps = 4
    batch = 1
    in_channels = 1
    out_channels= 1
    kernel_size = 2
    dilation = 1
    x = torch.randn(batch, in_channels, timesteps)
    c = torch.randn(batch, 2 * out_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation, use_film=True)
    y1 = conv(x, c, sequential=True)
    y2 = conv(x, c, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"


def test_forward_cond_film_mimo():
    print("Testing film conditional forward pass with Multiple-In-Multiple-Out, batch size 1")
    torch.manual_seed(42)
    timesteps = 4
    batch = 1
    in_channels = 3
    out_channels= 5
    kernel_size = 2
    dilation = 1
    x = torch.randn(batch, in_channels, timesteps)
    c = torch.randn(batch, 2 * out_channels, timesteps)
    conv = Convolution(in_channels, out_channels, kernel_size, dilation=dilation, use_film=True)
    y1 = conv(x, c, sequential=True)
    y2 = conv(x, c, sequential=False)
    assert torch.allclose(y1, y2, atol=1e-6, rtol=1e-5), \
        f"Outputs must match \n ext: {y1} \n ref: {y2}"


