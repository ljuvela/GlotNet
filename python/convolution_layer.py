import torch
import glotnet.cpp_extensions as ext
from glotnet.convolution import Convolution

class ConvolutionLayerFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight_conv, bias_conv, weight_out, bias_out, dilation, activation, use_output_transform):

        input = input.contiguous()
        # weights (OUT_CH, IN_CH, W) -> (W, IN_CH, OUT_CH)
        weight_conv = weight_conv.permute(2, 1, 0).contiguous()
        weight_out = weight_out.permute(2, 1, 0).contiguous()
        bias_conv = bias_conv.contiguous()
        bias_out = bias_out.contiguous()
        ctx.save_for_backward(input, weight_conv, bias_conv,
                              weight_out, bias_out,
                              torch.tensor(dilation))
        
        print(f"Conv weight shape {weight_conv.shape}")

        training = False
        output, skip = ext.convolution_layer_forward(
            input, weight_conv, bias_conv, weight_out, bias_out,
            training, dilation, use_output_transform, activation)
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
                 training=True,
                 activation="gated",
                 use_output_transform=True,
                 ):
        super().__init__()

        self.training = training
        residual_channels = out_channels
        self.activation = activation
        self.activation_fun, self.channel_mul = self._parse_activation(activation)
        self.conv = Convolution(
            in_channels=in_channels,
            out_channels=self.channel_mul*residual_channels,
            kernel_size=kernel_size, dilation=dilation, bias=bias, device=device, dtype=dtype,
            causal=causal, training=training)
        self.out = Convolution(
            in_channels=residual_channels,
            out_channels=residual_channels,
            kernel_size=1, dilation=1, bias=bias, device=device, dtype=dtype,
            training=training)
        self.residual_channels = residual_channels
        self.dilation = dilation
        self.use_output_transform = use_output_transform
        
    def _parse_activation(self, activation):
        activations = {
            "gated" : ((torch.tanh, torch.sigmoid), 2),
            "tanh" : (torch.tanh, 1),
            "linear" : (torch.nn.Identity(), 1)
        }
        activation_fun, channel_mul = activations.get(activation, (None, None))
        if channel_mul is None:
            raise NotImplementedError
        return activation_fun, channel_mul

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
            
            x = self.conv(input)
            if self.channel_mul == 2:
                R = self.residual_channels
                x = self.activation_fun[0](x[:, :R, :]) * self.activation_fun[1](x[:, R:, :])
            else:
                x = self.activation_fun(x)
            skip = x
            if self.use_output_transform:
                output = self.out(x)
            else:
                output = x
            return output, skip
        else:
            output, skip = ConvolutionLayerFunction.apply(input, self.conv.weight, self.conv.bias,
            self.out.weight, self.out.bias, self.dilation, self.activation, self.use_output_transform)
            return output, skip


