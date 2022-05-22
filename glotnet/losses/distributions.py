import torch
from glotnet.model.convolution import Convolution
import glotnet.cpp_extensions as ext

class Distribution(torch.nn.Module):
    """ Base class for distributions """
    def __init__(self):
        super().__init__()

class GaussianDensity(Distribution):

    def __init__(self, num_bits=None, temperature=1.0):
        """
        Args:
            input_channels: number of input channels
            num_bits: number of bits used to calculate entropy floor
        """
        super().__init__()
        self.num_bits = num_bits
        self.temperature = temperature
        self.register_buffer('const', torch.tensor(2 * torch.pi).sqrt().log())

    def set_temperature(self, temp: float):
        self.temperature = temp

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

    def sample(self, params, use_extension=False):
        
        if use_extension:
            params = params.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            x, = ext.sample_gaussian(params.contiguous(), self.temperature)
            x = x.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        else:
            m = params[:, 0:1, :]
            log_s = params[:, 1:2, :]
            s = torch.exp(log_s)
            x = m + s * torch.randn_like(m) * self.temperature 

        return x



class Identity(Distribution):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
    
    def sample(self, x):
        return x