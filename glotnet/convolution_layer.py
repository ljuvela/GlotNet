import torch
from typing import Tuple
import glotnet.cpp_extensions as ext
from glotnet.convolution import Convolution

class ConvolutionLayerFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, weight_conv: torch.Tensor, bias_conv: torch.Tensor,
                weight_out: torch.Tensor, bias_out: torch.Tensor,
                weight_skip: torch.Tensor, bias_skip: torch.Tensor,
                dilation: int, activation: str, use_output_transform: bool,
                cond_input: torch.Tensor = None, time_major: bool = True
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Convolution Layer forward

        Args:
            input: tensor of shape (batch, channels, time) (default) or (batch, time, channels)
            weight_conv: Conv1d weight tensor
                shape = (2 * ch_out, ch_in, kernel_size)
            bias_conv: 
                shape = (ch_out,)
            dilation: dilation factor (int)
            cond_input: (default = None)
            time_major: 
                if True: input.shape == (batch, channels, time), (PyTorch default)
                else: input.shape == (batch, time, channels),

        Returns:
            output: layer output, shape = (batch, ch_out, timesteps)
            skip: skip output, shape = (batch, ch_out, timesteps)

            if use_output_transform == False 
                'output' and 'skip' are the same variable
        """
        ctx.time_major = time_major
        if ctx.time_major:
            input = input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            if cond_input is not None:
                cond_input = cond_input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
                cond_input = cond_input.contiguous()

        input = input.contiguous()
        ctx.save_for_backward(input, weight_conv, bias_conv,
                              weight_out, bias_out)
        ctx.dilation = dilation

        use_skips = weight_skip is not None

        training = False
        if cond_input is None:
            if use_skips:
                output, skip = ext.convolution_layer_skip_forward(
                    input, weight_conv, bias_conv, weight_out, bias_out, weight_skip, bias_skip,
                    training, dilation, use_output_transform, activation)
            else:
                output, = ext.convolution_layer_forward(
                    input, weight_conv, bias_conv, weight_out, bias_out,
                    training, dilation, use_output_transform, activation)
        else:
            if use_skips:
                output, skip = ext.convolution_layer_skip_cond_forward(
                    input, cond_input, weight_conv, bias_conv, weight_out, bias_out, weight_skip, bias_skip,
                    training, dilation, use_output_transform, activation)
            else:
                pass

        if ctx.time_major:
            output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
            if use_skips:
                skip = skip.permute(0, 2, 1) # (B, T, C) -> (B, C, T)

        if use_skips:
            return output, skip
        else:
            return output

    @staticmethod
    def backward(self, d_output, d_skip):
        raise NotImplementedError("Backward function not implemented for sequential processing")


class ConvolutionLayer(torch.nn.Module):
    """
    Wavenet Convolution Layer (also known as Residual Block)

    Uses a gated activation and a 1x1 output transformation by default
    
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation=1,
                 bias=True, device=None, dtype=None,
                 causal=True,
                 training=True,
                 activation="gated",
                 use_output_transform=True,
                 cond_channels=None,
                 skip_channels=None,
                 ):
        super().__init__()

        self.training = training
        residual_channels = out_channels
        self.activation = activation
        self.activation_fun, self.channel_mul = self._parse_activation(activation)
        self.use_output_transform = use_output_transform
        self.use_conditioning = cond_channels is not None
        self.cond_channels = cond_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation = dilation
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
        if self.skip_channels is not None:
            self.skip = Convolution(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=1, dilation=1, bias=bias, device=device, dtype=dtype,
                training=training)
        if self.use_conditioning:
            self.cond_1x1 = torch.nn.Conv1d(cond_channels, self.channel_mul * residual_channels,
                kernel_size=1, bias=False, device=device, dtype=dtype)

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

    def forward(self, input, cond_input=None, sequential=False):
        """ 
        Args:
            input, torch.Tensor of shape (batch_size, in_channels, timesteps)
            sequential (optional), 
                if True, use CUDA compatible parallel implementation
                if False, use custom C++ sequential implementation 

        Returns:
            output, torch.Tensor of shape (batch_size, out_channels, timesteps)
            skip, torch.Tensor of shape (batch_size, out_channels, timesteps)
        
        """

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        if self.use_conditioning:
            c = self.cond_1x1(cond_input)
        else:
            c = None

        if sequential:
            if self.skip_channels is not None:
                output, skip = ConvolutionLayerFunction.apply(
                    input, self.conv.weight, self.conv.bias,
                    self.out.weight, self.out.bias,
                    self.skip.weight, self.skip.bias,
                    self.dilation, self.activation,
                    self.use_output_transform, c)
                return output, skip
            else:
                output = ConvolutionLayerFunction.apply(
                    input, self.conv.weight, self.conv.bias,
                    self.out.weight, self.out.bias,
                    None, None,
                    self.dilation, self.activation,
                    self.use_output_transform, c)
                return output
        else:
            x = self.conv(input, cond_input=c)
            if self.channel_mul == 2:
                R = self.residual_channels
                x = self.activation_fun[0](x[:, :R, :]) * self.activation_fun[1](x[:, R:, :])
            else:
                x = self.activation_fun(x)
            
            if self.skip_channels is not None:
                skip = self.skip(x)
            
            if self.use_output_transform:
                output = self.out(x)
            else:
                output = x

            if self.skip_channels is None:
                return output 
            else:
                return output, skip

