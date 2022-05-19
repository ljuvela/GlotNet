import torch
from glotnet.model.convolution import Convolution
import glotnet.cpp_extensions as ext
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

    def sample(self, params, temperature=1.0, use_extension=False):
        
        if use_extension:
            params = params.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            x, = ext.sample_gaussian(params.contiguous(), temperature)
            x = x.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        else:
            m = params[:, 0:1, :]
            log_s = params[:, 1:2, :]
            s = torch.exp(log_s)
            x = m + s * torch.randn_like(m) * temperature 

        return x

