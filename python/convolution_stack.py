import torch
import glotnet.cpp_extensions as ext
from glotnet.convolution_layer import ConvolutionLayer

class ConvolutionStackFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input, weights_conv, biases_conv, weights_out, biases_out, dilations, activation, use_residual):

        num_layers = len(dilations)

        input = input.contiguous()
        # weights (OUT_CH, IN_CH, W) -> (W, IN_CH, OUT_CH)
        weights_conv = [w.permute(2, 1, 0).contiguous() for w in weights_conv]
        weights_out = [w.permute(2, 1, 0).contiguous() for w in weights_out]
        biases_conv = [b.contiguous() for b in biases_conv]
        biases_out = [b.contiguous() for b in biases_out]
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
        raise NotImplementedError

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
                 ):
        super().__init__()

        self.training = training
        self.channels = channels
        self.activation = activation
        self.dilations = dilations
        self.use_residual = use_residual

        self.layers = torch.nn.ModuleList()
        for d in dilations:
            self.layers.append(
                ConvolutionLayer(
                    in_channels=channels, out_channels=channels,
                    kernel_size=kernel_size, dilation=d, bias=bias, device=device, dtype=dtype,
                    causal=causal,
                    training=training,
                    activation=activation,
                    use_output_transform=True
                )
            )


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
            conv_weights = []
            conv_biases = []
            out_weights = []
            out_biases = []
            for layer in self.layers:
                conv_weights.append(layer.conv.weight)
                conv_biases.append(layer.conv.bias)
                out_weights.append(layer.out.weight)
                out_biases.append(layer.out.bias)
            output, skips = ConvolutionStackFunction.apply(
                input, conv_weights, conv_biases,
                out_weights, out_biases, self.dilations, self.activation, self.use_residual)
            return output, skips

