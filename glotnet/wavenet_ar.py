import torch
import glotnet.cpp_extensions as ext
from glotnet.convolution_layer import ConvolutionLayer
from glotnet.convolution_stack import ConvolutionStack

class WaveNetARFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                input_weight, input_bias,
                output_weights, output_biases,
                dilations, use_residual, activation):

        num_layers = len(dilations)

        import ipdb; ipdb.set_trace()
        # Transpose input (batch, channels, time) -> (batch, time, channels)
        input = input.permute(0, 2, 1).contiguous()

        # TODO:
        # ctx.save_for_backward(...)

        training = False
        output, = ext.wavenet_ar_forward(input,
            stack_weights_conv, stack_biases_conv,
            stack_weights_out, stack_biases_out,
            input_weight, input_bias,
            output_weights, output_biases,
            dilations, training, use_residual, activation)

        return output

    def backward(self, d_output, d_skip):
        raise NotImplementedError

class WaveNetARCondFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, cond_input,
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                input_weight, input_bias,
                output_weights, output_biases,
                dilations, use_residual, activation):

        num_layers = len(dilations)

        input = input.contiguous()

        # TODO:
        # ctx.save_for_backward(...)
        
        training = False
        output, = ext.wavenet_ar_cond_forward(input, cond_input,
            stack_weights_conv, stack_biases_conv,
            stack_weights_out, stack_biases_out,
            input_weight, input_bias,
            output_weights, output_biases,
            dilations, training, use_residual, activation)

        return output

    def backward(self, d_output, d_skip):
        raise NotImplementedError

class WaveNetAR(torch.nn.Module):
    """
    WaveNetAR

    Autoregressive WaveNet

    """

    def __init__(self, input_channels, output_channels, residual_channels, kernel_size, dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                 bias=True, device=None, dtype=None,
                 causal=True,
                 training=True,
                 activation="gated",
                 use_residual=True,
                 use_1x1_block=True,
                 cond_channels=0,
                 ):
        super().__init__()

        self.training = training
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.residual_channels = residual_channels
        self.skip_channels = residual_channels
        self.cond_channels = cond_channels
        self.use_conditioning = cond_channels > 0
        self.activation = activation
        self.dilations = dilations
        self.use_residual = use_residual
        self.num_layers = len(dilations)

        # Layers
        self.input = ConvolutionLayer(
            in_channels=self.input_channels,
            out_channels=self.residual_channels,
            kernel_size=1, activation="tanh", use_output_transform=False)
        self.stack = ConvolutionStack(residual_channels, kernel_size, dilations=dilations,
                 activation=self.activation, use_residual=True, cond_channels=cond_channels)
        self.output1 = ConvolutionLayer(
            in_channels=self.skip_channels * self.num_layers,
            out_channels=self.residual_channels,
            kernel_size=1, activation="tanh", use_output_transform=False)
        self.output2 = ConvolutionLayer(
            in_channels=self.residual_channels,
            out_channels=self.output_channels,
            kernel_size=1, activation="linear", use_output_transform=False)

    @property 
    def output_weights(self):
        return [self.output1.conv.weight, self.output2.conv.weight]

    @property
    def output_biases(self):
        return [self.output1.conv.bias, self.output2.conv.bias]

    def forward(self, cond_input=None, timesteps=None, use_cpu=False):
        """ 
        Args:
            cond_input (optional),
                torch.Tensor of shape (batch_size, cond_channels, timesteps)
            use_cpu (optional), 
                if False, use CUDA compatible implementation
                if True, use custom C++ sequential implementation 

        Returns:
            output, torch.Tensor of shape (batch_size, output_channels, timesteps)
     
        """

        import ipdb; ipdb.set_trace()
        if cond_input is None and timesteps is None:
            raise RuntimeError("Either 'cond_input' or 'timesteps' must be specified")

        if cond_input is not None and timesteps is not None:
            raise RuntimeError("'cond_input' and 'timesteps' cannot both be specified")

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        if use_cpu:
            if cond_input is None:
                output = WaveNetARFunction.apply(
                    input,
                    self.stack.weights_conv, self.stack.biases_conv,
                    self.stack.weights_out, self.stack.biases_out,
                    self.input.conv.weight, self.input.conv.bias,
                    self.output_weights, self.output_biases,
                    self.dilations, self.use_residual, self.activation
                )
            else:
                conditioning = self.stack.project_conditioning(cond_input)
                output = WaveNetARCondFunction.apply(
                    input, conditioning,
                    self.stack.weights_conv, self.stack.biases_conv,
                    self.stack.weights_out, self.stack.biases_out,
                    self.input.conv.weight, self.input.conv.bias,
                    self.output_weights, self.output_biases,
                    self.dilations, self.use_residual, self.activation
                )
            return output
        else:
            # This implementation is just for checking correctness, 
            # never use this for processing
            if timesteps is None:
                timesteps = cond_input.size(-1)
            if cond_input is None:
                batch_size = 1
            else:
                batch_size = cond_input.size(0)

            with torch.no_grad():
                
                import ipdb; ipdb.set_trace()
                context = torch.zeros(batch_size, self.input_channels, timesteps)
                # Loop over time
                for t in range(timesteps):
                    x, _ = self.input(context)
                    _, skips = self.stack(x, cond_input)
                    x = torch.cat(skips, dim=1)
                    x, _ = self.output1(x)
                    x, _ = self.output2(x)

                    # Update context circular buffer
                    context = torch.roll(context, 1, dim=-1)
                    context[:, :, -1] = x
            
            return context

