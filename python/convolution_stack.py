import torch
import glotnet.cpp_extensions as ext
from glotnet.convolution_layer import ConvolutionLayer

class ConvolutionStackFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights_conv, biases_conv, weights_out, biases_out, dilations, activation, use_residual):

        num_layers = len(dilations)

        input = input.contiguous()
        ctx.save_for_backward(input, weights_conv, biases_conv,
                              weights_out, biases_out,
                              torch.tensor(dilations))
        
        training = False
        output, skips = ext.convolution_stack_forward(
            input, weights_conv, biases_conv, weights_out, biases_out,
            dilations, training, use_residual, activation)

        skips = skips.chunk(num_layers, dim=1)
        return output, skips

    def backward(self, d_output, d_skip):
        raise NotImplementedError("Backward function not implemented for sequential processing")

class ConvolutionStackCondFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, cond_input, weights_conv, biases_conv, weights_out, biases_out, dilations, activation, use_residual):

        num_layers = len(dilations)

        cond_input = cond_input.contiguous()
        input = input.contiguous()
        # ctx.save_for_backward(input, cond_input, weights_conv, biases_conv,
        #                       weights_out, biases_out,
        #                       torch.tensor(dilations))
        
        training = False
        output, skips = ext.convolution_stack_cond_forward(
            input, cond_input, weights_conv, biases_conv, weights_out, biases_out,
            dilations, training, use_residual, activation)

        skips = skips.chunk(num_layers, dim=1)
        return output, skips

    def backward(self, d_output, d_skip):
        raise NotImplementedError("Backward function not implemented for sequential processing")


class ConvolutionStack(torch.nn.Module):
    """
    Wavenet Convolution Stack

    Uses a gated activation and residual connections by default
    """

    def __init__(self, channels, kernel_size, dilations=[1], bias=True, device=None, dtype=None,
                 causal=True,
                 training=True,
                 activation="gated",
                 use_residual=True,
                 use_1x1_block_out=True,
                 cond_channels=0,
                 ):
        super().__init__()

        self.training = training
        self.channels = channels
        self.activation = activation
        self.dilations = dilations
        self.use_residual = use_residual
        self.use_1x1_block_out = use_1x1_block_out
        self.cond_channels = cond_channels
        self.num_layers = len(dilations)
        self.use_conditioning = cond_channels > 0

        self.layers = torch.nn.ModuleList()
        for i, d in enumerate(dilations):
            use_output_transform = self.use_1x1_block_out
            # Always disable output 1x1 for last layer
            if i == self.num_layers - 1:
                use_output_transform = False
            # Add ConvolutionLayer to Stack
            self.layers.append(
                ConvolutionLayer(
                    in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, dilation=d, bias=bias, device=device, dtype=dtype,
                    causal=causal,
                    training=training,
                    activation=activation,
                    use_output_transform=use_output_transform,
                    cond_channels=self.cond_channels
                )
            )

    @property
    def weights_conv(self):
        return [layer.conv.weight for layer in self.layers]
    
    @property
    def biases_conv(self):
        return [layer.conv.bias for layer in self.layers]

    @property
    def weights_out(self):
        return [layer.out.weight for layer in self.layers]

    @property
    def biases_out(self):
        return [layer.out.bias for layer in self.layers]

    def forward(self, input, cond_input=None, sequential=False):
        """ 
        Args:
            input, torch.Tensor of shape (batch_size, channels, timesteps)
            sequential (optional), 
                if True, use CUDA compatible parallel implementation
                if False, use custom C++ sequential implementation 

        Returns:
            output, torch.Tensor of shape (batch_size, channels, timesteps)
            skips, list of torch.Tensor of shape (batch_size, out_channels, timesteps)
        
        """

        if cond_input is not None and not self.use_conditioning:
            raise RuntimeError("Module has not been initialized to use conditioning, but conditioning input was provided at forward pass")

        if sequential:
            if cond_input is None:
                # no conditioning
                output, skips = ConvolutionStackFunction.apply(
                    input,
                    self.weights_conv, self.biases_conv,
                    self.weights_out, self.biases_out,
                    self.dilations, self.activation, self.use_residual)
            else:
                # project conditioning for each layer
                c_list = []
                for layer in self.layers:
                    c = layer.cond_1x1(cond_input)
                    c_list.append(c)
                cond_inputs = torch.cat(c_list, dim=1)
                output, skips = ConvolutionStackCondFunction.apply(
                    input, cond_inputs,
                    self.weights_conv, self.biases_conv,
                    self.weights_out, self.biases_out,
                    self.dilations, self.activation, self.use_residual)
            return output, skips
        else:
            x = input
            skips = []
            for layer in self.layers:
                h = x
                x, s = layer(x, cond_input, sequential=False)
                x = x + h  # residual connection
                skips.append(s)
            return x, skips
