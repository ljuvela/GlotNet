import torch
import glotnet.cpp_extensions as ext
from glotnet.convolution_layer import ConvolutionLayer
from glotnet.convolution_stack import ConvolutionStack

class WaveNetFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input,
                stack_weights_conv, stack_biases_conv,
                stack_weights_out, stack_biases_out,
                input_weight, input_bias,
                output_weights, output_biases,
                dilations, training, use_residual, activation):

        num_layers = len(dilations)

        input = input.contiguous()

        # TODO:
        # ctx.save_for_backward(...)
        
        training = False
        output, = ext.wavenet_forward(input,
            stack_weights_conv, stack_biases_conv,
            stack_weights_out, stack_biases_out,
            input_weight, input_bias,
            output_weights, output_biases,
            dilations, training, use_residual, activation)

        return output

    def backward(self, d_output, d_skip):
        raise NotImplementedError

class WaveNet(torch.nn.Module):
    """
    Wavenet

    Full wavenet

    """

    def __init__(self, input_channels, output_channels, residual_channels, kernel_size, dilations=[1, 2, 4, 8, 16, 32, 64, 128, 256],
                 bias=True, device=None, dtype=None,
                 causal=True,
                 training=True,
                 activation="gated",
                 use_residual=True,
                 use_1x1_block=True,
                 ):
        super().__init__()

        self.training = training
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.residual_channels = residual_channels
        self.skip_channels = residual_channels

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
                 activation=self.activation, use_residual=True)
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

    def forward(self, input, training=None):
        
        if training is not None:
            self.training = training

        if self.training:
            x = input
            x, _ = self.input(x)
            _, skips = self.stack(x)
            x = torch.cat(skips, dim=1)
            x, _ = self.output1(x)
            x, _ = self.output2(x)
            return x
        else:
            assert input.size(0)
            output = WaveNetFunction.apply(
                input,
                self.stack.weights_conv, self.stack.biases_conv,
                self.stack.weights_out, self.stack.biases_out,
                self.input.conv.weight, self.input.conv.bias,
                self.output_weights, self.output_biases,
                self.dilations, training, self.use_residual, self.activation
            )
            return output

