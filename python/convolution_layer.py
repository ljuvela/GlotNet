import torch
import glotnet.cpp_extensions as ext
from glotnet.convolution import Convolution

class ConvolutionLayerFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight_conv, bias_conv, weight_out, bias_out, dilation):

        input = input.contiguous()
        # weights (OUT_CH, IN_CH, W) -> (W, CH_IN, CH_OUT)
        weight_conv = weight_conv.permute(2, 1, 0).contiguous()
        weight_out = weight_out.permute(2, 1, 0).contiguous()
        bias_conv = bias_conv.contiguous()
        bias_out = bias_out.contiguous()
        ctx.save_for_backward(input, weight_conv, bias_conv, weight_out, bias_out, dilation)
        
        training = False
        use_residual = True
        activation = "gated"
        output, skip = ext.convolution_layer_forward(
            input, weight_conv, bias_conv, weight_out, bias_out,
            training, dilation, use_residual, activation)
        import ipdb; ipdb.set_trace()
        return output, skip

    def backward(self, d_output, d_skip):
        raise NotImplementedError

class ConvolutionLayer(torch.nn.Module):
    """
    Wavenet Convolution Layer (also known as Residual Block)

    Uses a gated activation and residual connections by default
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1, bias=True, device=None, dtype=None,
                 causal=True,
                 training=True,):
        super().__init__()

        self.training = training
        residual_channels = out_channels
        self.conv = Convolution(in_channels=in_channels, out_channels=2*residual_channels, kernel_size=kernel_size, dilation=dilation, bias=bias, device=device, dtype=dtype, causal=causal, training=training)
        self.out = Convolution(in_channels=residual_channels, out_channels=residual_channels, kernel_size=1, dilation=1, bias=bias, device=device, dtype=dtype, training=training)
        self.residual_channels = residual_channels
        self.dilation = dilation

    def forward(self, input, training=None):
        """ 
        Args:
            input, torch.Tensor of shape (batch_size, in_channels, timesteps)
            training (optional), 
                if True, use CUDA compatible parallel implementation
                if False, use custom C++ sequential implementation 

        Returns:
            output, torch.Tensor of shape (batch_size, out_channels, timesteps)
            skip, torch.Tensor of shape (batch_size, out_channels, timesteps)
        
        """
        
        if training is not None:
            self.training = training

        if self.training:
            R = self.residual_channels
            x = self.conv(input)
            x = torch.tanh(x[:, :R, :]) * torch.sigmoid(x[:, R:, :])
            skip = x
            output = self.out(x) + input
            return output, skip
        else:
            output, skip = ConvolutionLayerFunction.apply(input, self.conv.weight, self.conv.bias,
            self.out.weight, self.out.bias, self.dilation)
            return output, skip


