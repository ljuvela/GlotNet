import torch
import glotnet.cpp_extensions as ext

class ConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, dilation):

        input = input.contiguous()
        ctx.save_for_backward(input, weight, bias)
        
        training = False
        output, = ext.convolution_forward(input, weight, bias, training, dilation)
        return output 

    def backward(self, d_output):
        raise NotImplementedError

class ConvolutionCondFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, cond_input, weight, bias, dilation):

        input = input.contiguous()
        ctx.save_for_backward(input, weight, bias)
        
        training = False
        output, = ext.convolution_cond_forward(input, cond_input, weight, bias, training, dilation)
        return output 

    def backward(self, d_output):
        raise NotImplementedError


class Convolution(torch.nn.Conv1d):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None,
                 causal=True,
                 training=True,):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, device, dtype)
        self.causal = causal
        self.training = training

    def forward(self, input, cond_input=None, training=None):
        
        if training is not None:
            self.training = training

        if cond_input is not None:
            assert cond_input.size(1) == self.out_channels, f"Cond input number of channels mismatch. Expected {self.out_channels}, got {cond_input.size(1)}"
            assert cond_input.size(2) == input.size(2), f"Mismatching timesteps, input has {input.size(2)}, cond_input has {cond_input.size(2)}" 

        if self.training:
            padding = self.dilation[0] * self.stride[0] * (self.kernel_size[0]-1)
            if padding > 0:
                input = torch.nn.functional.pad(input, (padding, 0))
            output = torch.nn.functional.conv1d(
                input, self.weight, bias=self.bias,
                stride=self.stride, padding=0,
                dilation=self.dilation, groups=self.groups)
            if cond_input is not None:
                output = output + cond_input
            return output
        else:
            if cond_input is None:
                output = ConvolutionFunction.apply(input, self.weight, self.bias, self.dilation[0])
            else:
                output = ConvolutionCondFunction.apply(input, cond_input, self.weight, self.bias, self.dilation[0])
            return output


