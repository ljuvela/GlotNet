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
                 ):
        super().__init__()

        self.training = training
        self.channels = channels
        self.activation = activation
        self.dilations = dilations
        self.use_residual = use_residual
        self.use_1x1_block_out = use_1x1_block_out
        self.num_layers = len(dilations)

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
                    use_output_transform=use_output_transform
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

    def forward(self, input, training=None):
        """ 
        Args:
            input, torch.Tensor of shape (batch_size, channels, timesteps)
            training (optional), 
                if True, use CUDA compatible parallel implementation
                if False, use custom C++ sequential implementation 

        Returns:
            output, torch.Tensor of shape (batch_size, channels, timesteps)
            skips, list of torch.Tensor of shape (batch_size, out_channels, timesteps)
        
        """
        
        if training is not None:
            self.training = training

        if self.training:
            x = input
            skips = []
            for layer in self.layers:
                h = x
                x, s = layer(x)
                x = x + h # residual connection
                skips.append(s)
            return x, skips
        else:
            output, skips = ConvolutionStackFunction.apply(
                input,
                self.weights_conv, self.biases_conv,
                self.weights_out, self.biases_out,
                self.dilations, self.activation, self.use_residual)
            return output, skips

