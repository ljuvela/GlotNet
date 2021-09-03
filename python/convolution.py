import torch
import glotnet.cpp_extensions as ext

class ConvolutionFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weight, bias, dilation):

        input = input.contiguous()
        # weight (OUT_CH, IN_CH, W) -> (W, IN_CH, OUT_CH)
        weight = weight.permute(2, 1, 0).contiguous()
        bias = bias.contiguous()
        ctx.save_for_backward(input, weight, bias)
        
        training = False
        output, = ext.convolution_forward(input, weight, bias, training, dilation)
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

    def forward(self, input, training=None):
        
        if training is not None:
            self.training = training

        if self.training:
            padding = self.dilation[0] * self.stride[0] * (self.kernel_size[0]-1)
            input = torch.nn.functional.pad(input, (padding, 0))
            output = torch.nn.functional.conv1d(input, self.weight, bias=self.bias, 
            stride=self.stride, padding=0, 
            dilation=self.dilation, groups=self.groups) 
            return output
        else:
            output = ConvolutionFunction.apply(input, self.weight, self.bias, self.dilation[0])
            return output


