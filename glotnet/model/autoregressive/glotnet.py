import torch
from typing import List
from glotnet.model.feedforward.wavenet import WaveNet
from glotnet.losses.distributions import Distribution, GaussianDensity, Identity
import glotnet.cpp_extensions as ext

import torch.nn.functional as F

class GlotNetAR(WaveNet):
    """ Autoregressive GlotNet """

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
            distribution: Distribution = GaussianDensity(), 
            hop_length:int=256,
            lpc_order:int=10,
            cond_net: torch.nn.Module = None,
            sample_after_filtering : bool = False,
            ):
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
                hop length: hop length AR parameters
                 
        """
        super().__init__(
            input_channels, output_channels,
            residual_channels, skip_channels,
            kernel_size, dilations,
            causal, activation,
            use_residual, cond_channels,
            cond_net=cond_net)

        self.hop_length = hop_length
        self.lpc_order = lpc_order
        self.sample_after_filtering = sample_after_filtering

        cond_channels_int = 0 if cond_channels is None else cond_channels
        self._impl = ext.GlotNetAR(
            input_channels, output_channels,
            residual_channels, skip_channels,
            cond_channels_int, kernel_size,
            activation, dilations,
            lpc_order, self.sample_after_filtering
        )
            
        self._validate_distribution(distribution)

    def _validate_distribution(self, distribution):

        if distribution is None:
            distribution = Identity()
        self.distribution = distribution

    def _pad(self, x: torch.Tensor = None):
        if x is not None:
            x = F.pad(x, (self.receptive_field, 0), mode='constant', value=0)
        return x

    def _pad_filter(self, a):
        # pad filter poly
        # a = F.pad(a, (self.receptive_field, 0), mode='replicate')

        # pad to impulse
        a = F.pad(a, (self.receptive_field, 0), mode='constant', value=0)
        a[:, 0, :self.receptive_field] = 1.0

        return a


    def inference(self, 
                  input: torch.Tensor,
                  a: torch.Tensor,
                  cond_input: torch.Tensor = None,
                  padding = True,
                  temperature: torch.Tensor = None
              ):
        """ GlotNet forward pass, fast inference implementation
        
        Args:
            input: shape (batch_size, channels, timesteps)
                excitation signal, set to zeros for full autoregressive operation
            a: LPC predictor polynomial, shape (batch_size, order+1, timesteps//hop_length)
            cond_input (optional)"
                 shape (batch_size, cond_channels, timesteps)

        Returns:
            output, torch.Tensor of shape (batch_size, output_channels, timesteps)
     
        """

        batch_size, channels, timesteps = input.size()

        if cond_input is not None:
            assert cond_input.size(-1) == input.size(-1)

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        if cond_input is None and self.use_conditioning:
            raise RuntimeError("Module has been initialized to use conditioning, but conditioning input was not provided at forward pass")

        if a.size(1) != self.lpc_order+1:
            raise RuntimeError(f"AR poly order must be {self.lpc_order+1}, got {a.size(1)}")

        # a = F.interpolate(a, size=(input.size(2)), mode='linear', align_corners=False)
        a = F.interpolate(a, size=(input.size(2)), mode='nearest')

        if temperature is None:
            temperature = torch.ones(batch_size, 1, timesteps, device=input.device, dtype=input.dtype)

        if padding:
            input = self._pad(input)
            a = self._pad_filter(a)
            cond_input = self._pad(cond_input)
            temperature = self._pad(temperature)

        if input.device != torch.device('cpu'):
            raise RuntimeError(f"Input tensor device must be cpu, got {input.device}")
        if cond_input is not None and cond_input.device != torch.device('cpu'):
            raise RuntimeError(f"Cond input device must be cpu, got {cond_input.device}")

        output = GlotNetARFunction.apply(
            self._impl, input, a,
            self.stack.weights_conv, self.stack.biases_conv,
            self.stack.weights_out, self.stack.biases_out,
            self.stack.weights_skip, self.stack.biases_skip,
            self.stack.weights_cond, self.stack.biases_cond,
            self.input.conv.weight, self.input.conv.bias,
            self.output_weights, self.output_biases,
            temperature,
            cond_input, self.receptive_field
        )
        if padding:
            output = output[:, :, self.receptive_field:]
        return output

    def forward(self, 
                input: torch.Tensor,
                a: torch.Tensor,
                cond_input: torch.Tensor = None,
                padding: bool = True,
                temperature: torch.Tensor = None
              ):
        """ GlotNet forward pass, slow reference implementation

        Args:
            input: shape (batch_size, channels, timesteps)
                excitation signal, set to zeros for full autoregressive operation
            a: LPC predictor polynomial, shape (batch_size, order+1, timesteps//hop_length)
            cond_input (optional)"
                 shape (batch_size, cond_channels, timesteps)
            padding: apply zero padding to input (disable for stateful operation)

        Returns:
            output, torch.Tensor of shape (batch_size, output_channels, timesteps)
     
        """

        if input.size(-1) > 100:
            raise RuntimeError("Too many time steps, use inference() for fast inference")

        if cond_input is not None:
            assert cond_input.size(-1) == input.size(-1)

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        if cond_input is None and self.use_conditioning:
            raise RuntimeError("Module has been initialized to use conditioning, but conditioning input was not provided at forward pass")

        if a.size(1) != self.lpc_order+1:
            raise RuntimeError(f"AR poly order must be {self.lpc_order+1}, got {a.size(1)}")

        # a = F.interpolate(a, size=(input.size(2)), mode='linear', align_corners=False)
        a = F.interpolate(a, size=(input.size(2)), mode='nearest')

        if padding:
            input = self._pad(input)
            a = self._pad_filter(a)
            cond_input = self._pad(cond_input)
            temperature = self._pad(temperature)

        num_frames = a.size(2)

        batch_size = input.size(0)
        channels = input.size(1)
        timesteps = input.size(2)

        context = torch.zeros(batch_size, self.input_channels, self.receptive_field)
        output = torch.zeros(batch_size, 1, timesteps)

        if temperature is None:
            temperature = torch.ones(batch_size, 1, timesteps, device=input.device, dtype=input.dtype)

        # zero pad conditioning input
        if cond_input is not None:
            cond_input = torch.cat([torch.zeros(batch_size, self.cond_channels, self.receptive_field), cond_input], dim=-1)

        # Loop over time
        for t in range(timesteps):
            if cond_input is None:
                cond_context = None
            else:
                cond_context = cond_input[:, :, t:t + self.receptive_field]

            params = super().forward(input=context, cond_input=cond_context)

            p_curr = context[:, 1:2, -1]

            if self.sample_after_filtering:
                params[:, 0:1, -1:] = p_curr + params[:, 0:1, -1:]
                x_curr = self.distribution.sample(
                    params[..., -1:], temperature=temperature[..., t:t+1])
                e_curr = x_curr - p_curr
            else:
                e_curr = self.distribution.sample(
                    params[..., -1:], temperature=temperature[..., t:t+1])
                x_curr = p_curr + e_curr

            # update output 
            output[:, :, t] = x_curr

            # advance context
            context = torch.roll(context, -1, dims=-1)

            # excitation
            context[:, 0:1, -1] = e_curr
            # sample
            context[:, 2:3, -1] = x_curr
            # prediction for next time step
            x_curr = context[:, 2:3, -self.lpc_order:]
            a1 = torch.flip(-a[:, 1:, t], [1]) # TODO: no flip?
            p_next = torch.sum(a1 * x_curr, dim=-1)
            context[:, 1:2, -1] = p_next

        # remove padding
        if padding:
            output = output[:, :, self.receptive_field:]

        return output



class GlotNetARFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, impl,
                input: torch.Tensor,
                a: torch.Tensor,
                stack_weights_conv: List[torch.Tensor], stack_biases_conv: List[torch.Tensor],
                stack_weights_out: List[torch.Tensor], stack_biases_out: List[torch.Tensor],
                stack_weights_skip: List[torch.Tensor], stack_biases_skip: List[torch.Tensor],
                stack_weights_cond: List[torch.Tensor], stack_biases_cond: List[torch.Tensor],
                input_weight: torch.Tensor, input_bias: torch.Tensor,
                output_weights: List[torch.Tensor], output_biases: List[torch.Tensor],
                temperature: torch.Tensor,
                cond_input: torch.Tensor = None,
                flush_samples: int = 0
                ):

        input = input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
        input = input.contiguous()

        a = a.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
        a = a.contiguous()

        if cond_input is not None:
            cond_input = cond_input.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            cond_input = cond_input.contiguous()

        if cond_input is None:
            impl.set_parameters(
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                stack_weights_skip, stack_biases_skip,
                input_weight, input_bias,
                output_weights, output_biases)

            impl.flush(flush_samples)
            output, = impl.forward(
                input, a, temperature)
        else:
            impl.set_parameters_conditional(
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                stack_weights_skip, stack_biases_skip,
                stack_weights_cond, stack_biases_cond,
                input_weight, input_bias,
                output_weights, output_biases)

            impl.flush(flush_samples)
            output, = impl.cond_forward(
                input, a, cond_input, temperature)

        output = output.permute(0, 2, 1) # (B, T, C) -> (B, C, T)

        return output

    def backward(self, d_output, d_skip):
        raise NotImplementedError
