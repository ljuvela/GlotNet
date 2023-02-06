import torch
from typing import Tuple, List, Union
import glotnet.cpp_extensions as ext
from .convolution import Convolution


class ConvolutionLayer(torch.nn.Module):
    """
    Wavenet Convolution Layer (also known as Residual Block)

    Uses a gated activation and a 1x1 output transformation by default

    """

    def __init__(self,
                 in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int = 1,
                 bias: bool = True,
                 device=None, dtype=None,
                 causal: bool = True,
                 activation: str = "gated",
                 use_output_transform: bool = True,
                 cond_channels: int = None,
                 skip_channels: int = None,
                 use_film: bool = False
                 ):
        super().__init__()

        residual_channels = out_channels
        self.activation = activation
        self.activation_funcs, self.channel_mul = self._parse_activation(activation)
        self.use_output_transform = use_output_transform
        self.use_conditioning = cond_channels is not None
        self.cond_channels = cond_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.dilation = dilation
        self.use_film = use_film
        self.conv = Convolution(
            in_channels=in_channels,
            out_channels=self.channel_mul * residual_channels,
            kernel_size=kernel_size, dilation=dilation, bias=bias,
            device=device, dtype=dtype, causal=causal, 
            use_film=self.use_film)
        self.out = Convolution(
            in_channels=residual_channels,
            out_channels=residual_channels,
            kernel_size=1, dilation=1, bias=bias, device=device, dtype=dtype)
        if self.skip_channels is not None:
            self.skip = Convolution(
                in_channels=residual_channels,
                out_channels=skip_channels,
                kernel_size=1, dilation=1, bias=bias, device=device, dtype=dtype)
        if self.use_conditioning:
            cond_mul = 2 if self.use_film else 1
            self.cond_1x1 = torch.nn.Conv1d(
                cond_channels, self.channel_mul * residual_channels * cond_mul,
                kernel_size=1, bias=True, device=device, dtype=dtype)
        # extension backend implementation
        self._impl = ext.ConvolutionLayer(
            in_channels, out_channels,
            0 if skip_channels is None else skip_channels,
            0 if cond_channels is None else cond_channels, 
            kernel_size, dilation,
            self.use_output_transform, self.use_film, 
            activation)
    
    @property
    def receptive_field(self):
        return self.conv.receptive_field + self.out.receptive_field

    activations = {
        "gated": ([torch.nn.Tanh(), torch.nn.Sigmoid()], 2),
        "tanh": ([torch.nn.Tanh()], 1),
        "linear": ([torch.nn.Identity()], 1)
    }
    def _parse_activation(self, activation) -> Tuple[List[torch.nn.Module], int]:
        activation_funcs, channel_mul = ConvolutionLayer.activations[activation]
        return activation_funcs, channel_mul


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
            raise RuntimeError("Module has not been initialized to use conditioning, \
                but conditioning input was provided at forward pass")

        if sequential:
            skip_weight = None if self.skip_channels is None else self.skip.weight
            skip_bias = None if self.skip_channels is None else self.skip.bias
            cond_weight = self.cond_1x1.weight if self.use_conditioning else None
            cond_bias = self.cond_1x1.bias if self.use_conditioning else None
            return ConvolutionLayerFunction.apply(
                self._impl, input,
                self.conv.weight, self.conv.bias,
                self.out.weight, self.out.bias,
                skip_weight, skip_bias,
                cond_weight, cond_bias,
                self.dilation, self.activation,
                self.use_output_transform, cond_input)
        else:
            return self._forward_native(input=input, cond_input=cond_input)


    def _forward_native(self, input: torch.Tensor, cond_input: Union[torch.Tensor, None]):
        """ Pure PyTorch implementation of forward pass

        Args:
            input, shape (batch_size, in_channels, timesteps)
            cond_input, shape (batch_size, cond_channels, timesteps)
        
        """
        c = self.cond_1x1(cond_input) if self.use_conditioning else None
        x = self.conv(input, cond_input=c)
        if self.channel_mul == 2:
            R = self.residual_channels
            x = self.activation_funcs[0](x[:, :R, :]) * self.activation_funcs[1](x[:, R:, :])
        else:
            x = self.activation_funcs[0](x)

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

class ConvolutionLayerFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, impl, input: torch.Tensor,
                weight_conv: torch.Tensor, bias_conv: torch.Tensor,
                weight_out: torch.Tensor, bias_out: torch.Tensor,
                weight_skip: Union[torch.Tensor, None],
                bias_skip: Union[torch.Tensor, None],
                weight_cond: Union[torch.Tensor, None],
                bias_cond: Union[torch.Tensor, None],
                dilation: int, activation: str, use_output_transform: bool,
                cond_input: torch.Tensor = None
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

        Returns:
            output: layer output, shape = (batch, ch_out, timesteps)
            skip: skip output, shape = (batch, ch_out, timesteps)

            if use_output_transform == False
                'output' and 'skip' are the same variable
        """

        input = input.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
        input = input.contiguous()
        if cond_input is not None:
            cond_input = cond_input.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
            cond_input = cond_input.contiguous()

        ctx.dilation = dilation

        use_skips = weight_skip is not None

        impl.set_convolution_weight(weight_conv)
        impl.set_convolution_bias(bias_conv)
        impl.set_output_weight(weight_out)
        impl.set_output_bias(bias_out)
        if weight_skip is not None:
            impl.set_skip_weight(weight_skip)
        if bias_skip is not None:
            impl.set_skip_bias(bias_skip)
        if weight_cond is not None:
            impl.set_cond_weight(weight_cond)
        if bias_cond is not None:
            impl.set_cond_bias(bias_cond)

        if cond_input is None:
            if use_skips:
                output, skip = impl.skip_forward(input)
            else:
                output, = impl.forward(input)
        else:
            if use_skips:
                output, skip = impl.skip_cond_forward(input, cond_input)
            else:
                raise NotImplementedError("Conditional forward without skip connections not implemented")

        output = output.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)
        if use_skips:
            skip = skip.permute(0, 2, 1)  # (B, T, C) -> (B, C, T)

        if use_skips:
            return output, skip
        else:
            return output

    @staticmethod
    def backward(self, d_output, d_skip):
        raise NotImplementedError("Backward function not implemented for sequential processing")

