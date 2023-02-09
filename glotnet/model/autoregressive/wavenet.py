import torch
from typing import List
from glotnet.model.feedforward.wavenet import WaveNet
from glotnet.losses.distributions import Distribution, GaussianDensity, Identity
import glotnet.cpp_extensions as ext


class WaveNetAR(WaveNet):
    """ Autoregressive WaveNet """

    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            residual_channels: int,
            skip_channels: int,
            kernel_size: int,
            dilations: List[int] = [1, 2, 4, 8, 16, 32, 64, 128, 256],
            causal: bool = True,
            activation: str = "gated",
            use_residual: bool = True,
            cond_channels: int = None,
            distribution: Distribution = GaussianDensity()):
        """
           Args:
                input_channels: input channels
                output_channels: output channels
                residual_channels: residual channels in main dilation stack
                skip_channels: skip channels going out of dilation stack
                kernel_size: dilated convolution kernel size
                dilations: dilation factor for each each layer, len determines number of layers
                causal: 
                activation: activation type for dilated conv, options ["gated", "tanh"]
                use_residual:
                cond_channels: number of conditioning channels
                distribution: 
                 
        """
        super().__init__(
            input_channels, output_channels,
            residual_channels, skip_channels,
            kernel_size, dilations,
            causal, activation,
            use_residual, cond_channels)
            
        self._validate_distribution(distribution)

    def _validate_distribution(self, distribution):

        if distribution is None:
            distribution = Identity()
        self.distribution = distribution

    def set_temperature(self, temperature):
        self.distribution.set_temperature(temperature)

    @property
    def temperature(self):
        return self.distribution.temperature

    def forward(self, input: torch.Tensor, cond_input: torch.Tensor = None,
                timesteps: int = None, use_extension: bool = True):
        """ 
        Args:
            input: shape (batch_size, channels, timesteps)
                excitation signal, set to zeros for full autoregressive operation
            cond_input (optional)"
                 shape (batch_size, cond_channels, timesteps)
            use_extension (optional), 
                if False, use slow native (CUDA compatible) implementation
                if True, use custom C++ sequential implementation 

        Returns:
            output, torch.Tensor of shape (batch_size, output_channels, timesteps)
     
        """

        if cond_input is not None:
            assert cond_input.size(-1) == input.size(-1)

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        if cond_input is None and self.use_conditioning:
            raise RuntimeError("Module has been initialized to use conditioning, but conditioning input was not provided at forward pass")


        if use_extension:
            time_major = True
            if input.device != torch.device('cpu'):
                raise RuntimeError(f"Input tensor device must be cpu, got {input.device}")
            if cond_input is not None and cond_input.device != torch.device('cpu'):
                raise RuntimeError(f"Cond input device must be cpu, got {cond_input.device}")
            output = WaveNetARFunction.apply(
                input,
                self.stack.weights_conv, self.stack.biases_conv,
                self.stack.weights_out, self.stack.biases_out,
                self.stack.weights_skip, self.stack.biases_skip,
                self.stack.weights_cond, self.stack.biases_cond,
                self.input.conv.weight, self.input.conv.bias,
                self.output_weights, self.output_biases,
                self.dilations, self.use_residual, self.activation,
                cond_input, time_major, self.temperature
            )
            return output
        else:
            return self._forward_native(input, cond_input)

    def _forward_native(self, input: torch.Tensor, cond_input: torch.Tensor=None):
        # This implementation is just for checking correctness, 
        # never use this for processing

        timesteps = input.size(-1)
        batch_size = input.size(0)

        context = torch.zeros(batch_size, self.input_channels, self.receptive_field)
        output = torch.zeros(batch_size, self.input_channels, timesteps)

        if cond_input is not None:
            cond_input = torch.cat([torch.zeros(batch_size, self.cond_channels, self.receptive_field), cond_input], dim=-1)

        # Loop over time
        for t in range(timesteps):
            if cond_input is None:
                cond_context = None
            else:
                cond_context = cond_input[:, :, t:t + self.receptive_field]
            x = super()._forward_native(input=context, cond_input=cond_context)
            x = self.distribution.sample(x)

            e_t = input[:, :, t]
            x_t = x[:, :, -1] + e_t
            output[:, :, t] = x_t
            context = torch.roll(context, -1, dims=-1)
            context[:, :, -1] = x_t

        return output

class WaveNetARFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input: torch.Tensor,
                stack_weights_conv: List[torch.Tensor], stack_biases_conv: List[torch.Tensor],
                stack_weights_out: List[torch.Tensor], stack_biases_out: List[torch.Tensor],
                stack_weights_skip: List[torch.Tensor], stack_biases_skip: List[torch.Tensor],
                stack_weights_cond: List[torch.Tensor], stack_biases_cond: List[torch.Tensor],
                input_weight: torch.Tensor, input_bias: torch.Tensor,
                output_weights: List[torch.Tensor], output_biases: List[torch.Tensor],
                dilations: List[int], use_residual: bool, activation: str,
                cond_input: torch.Tensor = None, time_major: bool = True,
                temperature: float = 1.0):

        num_layers = len(dilations)

        ctx.time_major = time_major
        if ctx.time_major:
            input = input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            input = input.contiguous()
            if cond_input is not None:
                cond_input = cond_input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
                cond_input = cond_input.contiguous()

        if cond_input is None:
            output, = ext.wavenet_ar_forward(
                input,
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                stack_weights_skip, stack_biases_skip,
                input_weight, input_bias,
                output_weights, output_biases,
                dilations, use_residual, activation, temperature)
        else:
            output, = ext.wavenet_ar_cond_forward(
                input, cond_input,
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                stack_weights_skip, stack_biases_skip,
                stack_weights_cond, stack_biases_cond,
                input_weight, input_bias,
                output_weights, output_biases,
                dilations, use_residual, activation, temperature)

        if ctx.time_major:
            output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)

        return output

    def backward(self, d_output, d_skip):
        raise NotImplementedError
