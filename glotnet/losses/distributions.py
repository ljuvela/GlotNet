import torch
import torch.nn.functional as F
import glotnet.cpp_extensions as ext
from glotnet.sigproc import mulaw

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

    cpp_name = 'gaussian'

    def __init__(self,
                 num_bits: int = None,
                 temperature: float = 1.0,
                 out_dim: int = 1,
                 entropy_floor: float = -7.0,
                 weight_nll: float = 1.0,
                 weight_entropy_penalty: float = 0.1):
        """
        Args:
            input_channels: number of input channels
            num_bits: number of bits used to calculate entropy floor
        """
        params_dim = 2 * out_dim
        super().__init__(params_dim=params_dim, out_dim=out_dim)
        self.num_bits = num_bits
        self.temperature = temperature
        self.entropy_floor = entropy_floor
        self.weight_nll = weight_nll
        self.weight_entropy_penalty = weight_entropy_penalty
        self.register_buffer('const', torch.tensor(2 * torch.pi).sqrt().log())

    def set_temperature(self, temp: float):
        self.temperature = temp

    def nll(self, x, params):
        """ Negative Log-Likelihood """
        # NLL
        m = params[:, 0:1, :]
        log_s = params[:, 1:2, :]
        s = torch.exp(log_s)

        # calculate entropy floor hinge regularizer
        entropy_floor = self.entropy_floor
        self.batch_penalty = log_s.clamp(max=entropy_floor).pow(2).sum()
        penalty_mask = log_s < entropy_floor
        self.batch_penalty = (log_s * penalty_mask).pow(2)
        self.batch_log_scale = log_s

        nll = 0.5 * (m - x).div(s).pow(2) + log_s + self.const
        self.batch_nll = nll
        return nll

    def shift_location(self, params, offset):
        """ Distribution of x + offset: the mean moves, the scale does not

        Args:
            params: (batch, 2, time), channels (m, log_s)
            offset: broadcastable to (batch, 1, time)
        """
        return torch.cat([params[:, 0:1, :] + offset, params[:, 1:, :]], dim=1)

    def forward(self, x, params):
        nll = self.nll(x, params)
        loss = self.weight_nll * nll.mean() 
        loss = loss + self.weight_entropy_penalty * self.batch_penalty.mean()
        return loss

    def sample(self, params, use_extension=False, temperature=None):
        
        if temperature is None:
            temperature = self.temperature * torch.ones_like(params[:, 0:1, :])

        if params.size(1) != self.params_dim:
            raise ValueError(f'Expected params to have {self.params_dim} channels, got {params.size(1)}')
        if use_extension:
            params = params.permute(0, 2, 1) # (B, C, T) -> (B, T, C)
            x, = ext.sample_gaussian(params.contiguous(), self.temperature)
            x = x.permute(0, 2, 1) # (B, T, C) -> (B, C, T)
        else:
            m = params[:, 0:1, :]
            log_s = params[:, 1:2, :]
            s = torch.exp(log_s)
            x = m + s * torch.randn_like(m) * temperature 

        return x



class CategoricalDensity(Distribution):

    cpp_name = 'categorical'

    """ Categorical distribution over mu-law quantized sample values

        The model outputs one logit per quantization level
        (params_dim = 2**num_bits). Samples are returned as signal values
        in [-1, 1] (dequantized and expanded), so the autoregressive
        feedback stays a single float channel.
    """

    def __init__(self,
                 num_bits: int = 8,
                 temperature: float = 1.0,
                 weight_nll: float = 1.0):
        """
        Args:
            num_bits: quantization bits, mu = 2**num_bits - 1
            temperature: default sampling temperature (logits are divided by it)
            weight_nll: loss weight
        """
        mu = mulaw.mu_from_bits(num_bits)
        super().__init__(params_dim=mu + 1, out_dim=1)
        self.num_bits = num_bits
        self.mu = mu
        self.temperature = temperature
        self.weight_nll = weight_nll

    def set_temperature(self, temp: float):
        self.temperature = temp

    def _check_params(self, params: torch.Tensor):
        if params.size(1) != self.params_dim:
            raise ValueError(f'Expected params to have {self.params_dim} channels, got {params.size(1)}')

    def nll(self, x, params):
        """ Negative log-likelihood (in nats) of the quantized samples

        Args:
            x: signal, shape (batch, 1, time), values in [-1, 1]
            params: logits, shape (batch, 2**num_bits, time)

        Returns:
            nll, shape (batch, 1, time)
        """
        self._check_params(params)
        target = mulaw.encode(x[:, 0, :], self.mu)
        nll = F.cross_entropy(params, target, reduction='none').unsqueeze(1)
        self.batch_nll = nll
        return nll

    def forward(self, x, params):
        nll = self.nll(x, params)
        return self.weight_nll * nll.mean()

    def sample(self, params, use_extension=False, temperature=None):
        """ Sample signal values

        Args:
            params: logits, shape (batch, 2**num_bits, time)
            temperature: float or tensor broadcastable to (batch, 1, time);
                logits are divided by it, 0 gives the most likely value

        Returns:
            x, shape (batch, 1, time), values in [-1, 1]
        """
        self._check_params(params)
        if temperature is None:
            temperature = self.temperature
        temperature = torch.as_tensor(temperature, dtype=params.dtype, device=params.device)
        temperature = temperature.expand(params.size(0), 1, params.size(2))

        if use_extension:
            # seed drawn from torch's RNG, so torch.manual_seed makes it reproducible
            seed = int(torch.randint(0, 2**62, (1,)).item())
            x, = ext.sample_categorical(
                params.transpose(1, 2).contiguous(),       # (B, T, K)
                temperature.transpose(1, 2).contiguous(),  # (B, T, 1)
                self.num_bits, seed)
            return x.transpose(1, 2)  # (B, 1, T)

        logits = params / temperature.clamp(min=1e-6)
        q = torch.distributions.Categorical(logits=logits.transpose(1, 2)).sample()
        # exact argmax where temperature is zero
        greedy = temperature[:, 0, :] <= 0.0
        q = torch.where(greedy, params.argmax(dim=1), q)
        return mulaw.decode(q, self.mu).unsqueeze(1)


class Identity(Distribution):
    def __init__(self):
        super().__init__(params_dim=1, out_dim=1)

    def forward(self, x):
        return x
    
    def sample(self, x):
        return x


def set_cpp_distribution(impl, distribution: Distribution, output_channels: int):
    """ Select the sampler of a C++ autoregressive model (ext.WaveNetAR / ext.GlotNetAR)

    Only applied when the distribution has a C++ sampler and its parameter
    count matches the model outputs; other combinations remain valid for
    teacher-forced use and are rejected by check_cpp_distribution at inference.
    """
    name = getattr(distribution, 'cpp_name', None)
    if name is None or distribution.params_dim != output_channels:
        return
    num_bits = distribution.num_bits if name == 'categorical' else 8
    impl.set_distribution(name, num_bits)


def check_cpp_distribution(impl, distribution: Distribution, output_channels: int):
    """ Validate before C++ inference and seed the sampler from torch's RNG """
    name = getattr(distribution, 'cpp_name', None)
    if name is None:
        raise ValueError(
            f"{type(distribution).__name__} has no C++ sampler, "
            "inference() supports GaussianDensity and CategoricalDensity")
    if distribution.params_dim != output_channels:
        raise ValueError(
            f"{type(distribution).__name__} needs {distribution.params_dim} "
            f"output channels, model has {output_channels}")
    # torch.manual_seed makes inference reproducible
    impl.set_seed(int(torch.randint(0, 2**62, (1,)).item()))
