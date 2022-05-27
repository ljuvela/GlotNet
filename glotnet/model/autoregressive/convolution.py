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
            return ConvolutionFunctionAR.apply(input, self.weight, self.bias, self.dilation[0], cond_input)
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
    def forward(ctx, input, weight, bias, dilation, cond_input=None, time_major=True):
        """ Dilated covolution bindings forward pass

        Args:
            input: tensor of shape (batch, channels, time) (default) or (batch, time, channels)
            weight: Conv1d weight tensor, shape = (ch_out, ch_in, kernel_size)
            bias: Conv1d bias tensor, shape = (ch_out,)
            dilation: int type dilation factor
            cond_input: (default = None)
            time_major: 
                if True: input.shape == (batch, channels, time), (PyTorch default)
                else: input.shape == (batch, time, channels),

        """
        ctx.time_major = time_major
        if ctx.time_major:
            input = input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            if cond_input is not None:
                cond_input = cond_input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
                cond_input = cond_input.contiguous()
        input = input.contiguous()

        if cond_input is None:
            output, = ext.convolution_forward_ar(input, weight, bias, dilation)
        else:
            output, = ext.convolution_cond_forward_ar(input, cond_input, weight, bias, dilation)

        if ctx.time_major:
            output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        return output 

    def backward(self, d_output):
        raise NotImplementedError