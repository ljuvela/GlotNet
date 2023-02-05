import torch
from glotnet.model.feedforward.convolution import Convolution
import glotnet.cpp_extensions as ext

class ConvolutionAR(Convolution):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, bias=True, device=None, dtype=None):

        super().__init__(in_channels, out_channels, kernel_size,
                         stride=stride, padding=0, dilation=dilation,
                         groups=1, bias=bias, padding_mode='zeros',
                         device=device, dtype=dtype, causal=True)
        self.input_channels = in_channels
        self.output_channels = out_channels

    def forward(self, input: torch.Tensor, cond_input: torch.Tensor = None, use_extension=False) -> torch.Tensor:

        if cond_input is not None:
            assert cond_input.size(1) == self.out_channels, f"Cond input number of channels mismatch. Expected {self.out_channels}, got {cond_input.size(1)}"
            assert cond_input.size(2) == input.size(2), f"Mismatching timesteps, input has {input.size(2)}, cond_input has {cond_input.size(2)}" 

        if use_extension:
            return ConvolutionFunctionAR.apply(
                self._impl,
                input, cond_input,
                *self.parameters())
        else:
            return self._forward_native(input=input, cond_input=cond_input)


    def _forward_native(self, input: torch.Tensor, cond_input: torch.Tensor = None) -> torch.Tensor:
        """
        Args: input, shape (batch, channels, timesteps)
        """

        timesteps = input.size(-1)
        batch_size = input.size(0)

        context = torch.zeros(batch_size, self.input_channels, self.receptive_field)
        output = torch.zeros(batch_size, self.input_channels, timesteps)
        # Loop over time
        for t in range(timesteps):
            x = super()._forward_native(context, cond_input=cond_input)

            e_t = input[:, :, t]
            x_t = x[:, :, -1] + e_t
            output[:, :, t] = x_t
            context = torch.roll(context, -1, dims=-1)
            context[:, :, -1] = x_t

        return output


class ConvolutionFunctionAR(torch.autograd.Function):

    @staticmethod
    def forward(ctx, impl, input, cond_input=None, *params):
        """ Dilated covolution bindings forward pass

        Args:
            impl: cpp extension object
            input: tensor of shape (batch, channels, time) (default) or (batch, time, channels)
            cond_input: (default = None)
            params: packed parameter list

        """
        weight, bias = params

        input = input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
        input = input.contiguous()
        if cond_input is not None:
            cond_input = cond_input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            cond_input = cond_input.contiguous()

        conv = impl
        conv.set_weight(weight)
        conv.set_bias(bias)
        if cond_input is None:
            output, = conv.forward_ar(input)
        else:
            output, = conv.cond_forward_ar(input, cond_input)

        output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        return output 

    def backward(self, d_output):
        raise NotImplementedError