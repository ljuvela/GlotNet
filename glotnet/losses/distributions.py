import torch
import glotnet.cpp_extensions as ext

class Distribution(torch.nn.Module):
    """ Base class for distributions
        
        Args:
            params_dim
    """

    def __init__(self, params_dim: int, out_dim: int):
        """
        Args:
            params_dim: number of parameters per sample
            out_dim: number of output channels per sample
        """
        super().__init__()
        self.params_dim = params_dim
        self.out_dim = out_dim

    def nll(self, x: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """ Negative log-likelihood """
        raise NotImplementedError

    def sample(self, params, use_extension=False) -> torch.Tensor:
        raise NotImplementedError

class GaussianDensity(Distribution):

    def __init__(self, num_bits=None, temperature=1.0, out_dim=1):
        """
        Args:
            input_channels: number of input channels
            num_bits: number of bits used to calculate entropy floor
        """
        params_dim = 2 * out_dim
        super().__init__(params_dim=params_dim, out_dim=out_dim)
        self.num_bits = num_bits
        self.temperature = temperature
        self.register_buffer('const', torch.tensor(2 * torch.pi).sqrt().log())

    def set_temperature(self, temp: float):
        self.temperature = temp

    def nll(self, x, params):
        """
        
        """


        # NLL
        m = params[:, 0:1, :]
        log_s = params[:, 1:2, :]
        s = torch.exp(log_s)

        # calculate entropy floor hinge regularizer
        entropy_floor = -7.0
        self.batch_penalty = log_s.clamp(max=entropy_floor).pow(2).sum()
        penalty_mask = log_s < entropy_floor
        self.batch_penalty = (log_s * penalty_mask).pow(2)
        self.batch_log_scale = log_s

        nll = 0.5 * (m - x).div(s).pow(2) + log_s + self.const
        self.batch_nll = nll
        return nll

    def forward(self, x, params):
        nll = self.nll(x, params)
        return nll.mean() + self.batch_penalty.mean()

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