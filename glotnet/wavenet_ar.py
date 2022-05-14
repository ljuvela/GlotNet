import torch
from typing import List
import glotnet.cpp_extensions as ext
from glotnet.convolution_layer import ConvolutionLayer
from glotnet.convolution_stack import ConvolutionStack
from glotnet.wavenet import WaveNet

class WaveNetAR(WaveNet):
    """ Autoregressive WaveNet """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, cond_input=None, timesteps=None, use_cpu=True):
        """ 
        Args:
            cond_input (optional),
                torch.Tensor of shape (batch_size, cond_channels, timesteps)
            use_cpu (optional), 
                if False, use slow CUDA compatible implementation
                if True, use custom C++ sequential implementation 

        Returns:
            output, torch.Tensor of shape (batch_size, output_channels, timesteps)
     
        """

        if cond_input is None and timesteps is None:
            raise RuntimeError("Either 'cond_input' or 'timesteps' must be specified")

        if cond_input is not None and timesteps is not None:
            raise RuntimeError("'cond_input' and 'timesteps' cannot both be specified")

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        if use_cpu:
            output = WaveNetARFunction.apply(
                timesteps,
                self.stack.weights_conv, self.stack.biases_conv,
                self.stack.weights_out, self.stack.biases_out,
                self.stack.weights_skip, self.stack.biases_skip,
                self.stack.weights_cond, self.stack.biases_cond,
                self.input.conv.weight, self.input.conv.bias,
                self.output_weights, self.output_biases,
                self.dilations, self.use_residual, self.activation,
                cond_input
            )
            return output
        else:
            return self._forward_native(cond_input, timesteps)

    def _forward_native(self, cond_input=None, timesteps=None):
        # This implementation is just for checking correctness, 
        # never use this for processing
        if timesteps is None:
            timesteps = cond_input.size(-1)
        if cond_input is None:
            batch_size = 1
        else:
            batch_size = cond_input.size(0)
        with torch.no_grad():
            context = torch.zeros(batch_size, self.input_channels, timesteps)
            # Loop over time
            for t in range(timesteps):
                x = self.input(context)
                _, skips = self.stack(x, cond_input)
                x = torch.stack(skips, dim=0).sum(dim=0)
                x = self.output1(x)
                x = self.output2(x)

                x_t = x[:, :, t]

                # Update context circular buffer
                context = torch.roll(context, -1, dims=-1)
                context[:, :, -1] = x_t
        return context

class WaveNetARFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, timesteps: int,
                stack_weights_conv: List[torch.Tensor], stack_biases_conv: List[torch.Tensor],
                stack_weights_out: List[torch.Tensor], stack_biases_out: List[torch.Tensor],
                stack_weights_skip: List[torch.Tensor], stack_biases_skip: List[torch.Tensor],
                stack_weights_cond: List[torch.Tensor], stack_biases_cond: List[torch.Tensor],
                input_weight: torch.Tensor, input_bias: torch.Tensor,
                output_weights: List[torch.Tensor], output_biases: List[torch.Tensor],
                dilations: List[int], use_residual: bool, activation: str,
                cond_input: torch.Tensor = None, time_major: bool = True):

        num_layers = len(dilations)

        ctx.time_major = time_major
        if ctx.time_major and cond_input is not None:
            cond_input = cond_input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            cond_input = cond_input.contiguous()

        if cond_input is None:
            output, = ext.wavenet_ar_forward(timesteps,
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                stack_weights_skip, stack_biases_skip,
                input_weight, input_bias,
                output_weights, output_biases,
                dilations, use_residual, activation)
        else:
            output, = ext.wavenet_ar_cond_forward(cond_input,
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                stack_weights_skip, stack_biases_skip,
                stack_weights_cond, stack_biases_cond,
                input_weight, input_bias,
                output_weights, output_biases,
                dilations, use_residual, activation)


        if ctx.time_major:
            output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)

        return output

    def backward(self, d_output, d_skip):
        raise NotImplementedError
