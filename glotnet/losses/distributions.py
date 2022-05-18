import torch
from glotnet.model.convolution import Convolution

class GaussianDensity(torch.nn.Module):

    def __init__(self, num_bits=None):
        """
        Args:
            input_channels: number of input channels
            num_bits: number of bits used to calculate entropy floor
        """
        super().__init__()
        self.num_bits = num_bits

        self.register_buffer('const', torch.tensor(2 * torch.pi).sqrt().log())

    def nll(self, x, params):
        """
        
        """
        # TODO: calculate entropy floor hinge regularizer


        #

        # NLL
        m = params[:, 0:1, :]
        log_s = params[:, 1:2, :]
        s = torch.exp(log_s)

        nll = 0.5 * (m - x).div(s).pow(2) + log_s + self.const
        return nll

    def forward(self, x, params):
        return self.nll(x, params)

    def sample(self, params):
        pass

