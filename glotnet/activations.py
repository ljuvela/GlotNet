import torch
import glotnet.cpp_extensions as ext
from glotnet.convolution import Convolution

class ConvolutionLayerFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor, 
                activation: str, time_major: bool = True
                ) -> torch.Tensor:

        ctx.time_major = time_major
        if ctx.time_major:
            input = input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
        
        input = input.contiguous()
        ctx.save_for_backward(input)
        ctx.activation_type = activation

        output, = ext.activations_forward(input, activation)

        if ctx.time_major:
            output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)

        return output 

    @staticmethod
    def backward(ctx, d_output):
        raise NotImplementedError("Backward function not implemented for sequential processing")


def _gated_activation(x: torch.Tensor) -> torch.Tensor:

    assert x.size(1) % 2 == 0, f"Input must have an even number of channels, shape was {x.shape}"
    half = x.size(1) // 2
    return torch.tanh(x[:, :half, :]) * torch.sigmoid(x[:, half:, :])

class Activation(torch.nn.Module):
    """ Activation class """

    def __init__(self, activation="gated"):
        super().__init__()
        self.activation_str = activation
        if activation == "gated":
            self.activation_func = _gated_activation
        elif activation == "tanh":
            self.activation_func = torch.tanh
        elif activation == "linear":
            self.activation_func = torch.nn.Identity()

    def forward(self, input, use_extension=True):

        if use_extension:
            return ConvolutionLayerFunction.apply(input, self.activation_str)
        else:
            return self.activation_func(input)
