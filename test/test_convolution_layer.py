import torch

from glotnet.convolution_layer import ConvolutionLayer


# def test_forward1():
if __name__ == "__main__":
    torch.manual_seed(42)
    timesteps = 100
    batch = 1 
    in_channels = 1
    out_channels= 1
    kernel_size = 50
    dilation = 1
    x = torch.randn(batch, in_channels, timesteps)
    conv = ConvolutionLayer(in_channels, out_channels, kernel_size, dilation=dilation)
    y1, s1 = conv(x, training=True)
    y2, s2 = conv(x, training=False)
    # assert torch.allclose(y1, y2)



# if __name__ == "__main__":
    # print("Testing forward pass with SISO, batch size 1")
    # test_forward1()
    # print("   ok!")




