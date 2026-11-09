import torch
import torch.nn.functional as F
import glotnet.cpp_extensions as ext
from glotnet.losses import location_scale
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


class LocationScaleMixture(Distribution):
    """ Mixture of location-scale densities (Gaussian, Laplace, logistic)

        Parameter channels, K = len(components):
            shared_loc=False: [logits (K), locs (K), log_scales (K)], params_dim 3K
            shared_loc=True:  [logits (K), loc (1),  log_scales (K)], params_dim 2K + 1

        Components may mix families, e.g. ('gaussian', 'laplace'). With
        discretize_bits set, the likelihood is the probability mass of the
        linear bin of the target (2**bits levels on [-1, 1], open edge bins,
        as in PixelCNN++ / Parallel WaveNet) and samples are snapped to the bins.

        Entropy floor: a hinge on each component's entropy at the entropy of
        a uniform density over one bin of an entropy_floor_bits signal, i.e.
        each family gets the log-scale floor with that entropy.
    """

    def __init__(self,
                 components=('gaussian', 'laplace'),
                 shared_loc: bool = False,
                 discretize_bits: int = None,
                 entropy_floor_bits: int = 16,
                 temperature: float = 1.0,
                 weight_nll: float = 1.0,
                 weight_entropy_penalty: float = 0.1):
        """
        Args:
            components: family name per component ('gaussian', 'laplace', 'logistic')
            shared_loc: one location for all components
            discretize_bits: None for a continuous density, else the bit depth
                of the linear discretization of [-1, 1]
            entropy_floor_bits: bit depth defining the entropy floor (None: no floor)
            temperature: default sampling temperature (scales the component scales)
            weight_nll, weight_entropy_penalty: loss weights
        """
        components = tuple(components)
        if len(components) == 0:
            raise ValueError("LocationScaleMixture needs at least one component")
        families = [location_scale.get_family(name) for name in components]
        if discretize_bits is not None and not 1 <= discretize_bits <= 24:
            raise ValueError(f"discretize_bits must be in [1, 24], got {discretize_bits}")
        K = len(components)
        super().__init__(params_dim=(2 * K + 1) if shared_loc else 3 * K, out_dim=1)
        self.components = components
        self.num_components = K
        self.shared_loc = shared_loc
        self.discretize_bits = discretize_bits
        self.entropy_floor_bits = entropy_floor_bits
        self.temperature = temperature
        self.weight_nll = weight_nll
        self.weight_entropy_penalty = weight_entropy_penalty

        # components grouped by family: one call per family, then reorder
        self._groups = []  # (family, component indices)
        for name in dict.fromkeys(components):
            idx = [k for k, c in enumerate(components) if c == name]
            self._groups.append((location_scale.get_family(name), idx))
        order = torch.tensor([k for _, idx in self._groups for k in idx])
        self.register_buffer('_inverse_order', torch.argsort(order), persistent=False)
        self.register_buffer('_family_of_component', torch.tensor(
            [g for g, (_, idx) in enumerate(self._groups) for _ in idx])[torch.argsort(order)],
            persistent=False)

        if entropy_floor_bits is None:
            floor = torch.full((K,), -float('inf'))
        else:
            h = location_scale.uniform_bin_entropy(entropy_floor_bits)
            floor = torch.tensor([f.log_scale_from_entropy(h) for f in families])
        self.register_buffer('log_scale_floor', floor.view(1, K, 1), persistent=False)

    # --- parameters ---

    def _check_params(self, params):
        if params.size(1) != self.params_dim:
            raise ValueError(f'Expected params to have {self.params_dim} channels, got {params.size(1)}')

    def split(self, params):
        """ (logits, locs, log_scales), each (batch, K, time); a shared
            location is expanded to K channels """
        self._check_params(params)
        K = self.num_components
        logits = params[:, :K]
        if self.shared_loc:
            locs = params[:, K:K + 1].expand(-1, K, -1)
            log_scales = params[:, K + 1:]
        else:
            locs = params[:, K:2 * K]
            log_scales = params[:, 2 * K:]
        return logits, locs, log_scales

    def shift_location(self, params, offset):
        """ Distribution of x + offset: every location moves, scales and weights do not """
        self._check_params(params)
        K = self.num_components
        n_loc = 1 if self.shared_loc else K
        return torch.cat([params[:, :K], params[:, K:K + n_loc] + offset,
                          params[:, K + n_loc:]], dim=1)

    def initial_bias(self, log_scale_range=(-5.0, -2.0)):
        """ Output-layer bias: zero logits and locations; log scales spread
            over the components of each family, at the same entropies for
            every family (the entropies of Gaussians with log scales in
            log_scale_range), so no family starts wider than another """
        K = self.num_components
        gaussian = location_scale.get_family('gaussian')
        h_lo, h_hi = (gaussian.entropy(v) for v in log_scale_range)
        log_scales = torch.zeros(K)
        for family, idx in self._groups:
            n = len(idx)
            h = torch.linspace(h_lo, h_hi, n) if n > 1 else torch.tensor([(h_lo + h_hi) / 2])
            log_scales[idx] = family.log_scale_from_entropy(h)
        n_loc = 1 if self.shared_loc else K
        return torch.cat([torch.zeros(K), torch.zeros(n_loc), log_scales])

    # --- per-component evaluation ---

    def _per_component(self, fn, locs, log_scales, *args):
        """ fn(family, *args, loc, log_scale) per family group -> (batch, K, time) """
        out = [fn(family, *args, locs[:, idx], log_scales[:, idx])
               for family, idx in self._groups]
        return torch.cat(out, dim=1).index_select(1, self._inverse_order)

    def bin_edges(self, x):
        """ Edges (lo, hi) of the discretization bin of each value; values
            beyond [-1, 1] fall into the open edge bins """
        n = 2 ** self.discretize_bits - 1  # number of steps between levels
        step = 2.0 / n
        q = torch.round((x.clamp(-1.0, 1.0) + 1.0) / step)
        center = q * step - 1.0
        inf = torch.full_like(x, float('inf'))
        lo = torch.where(q <= 0, -inf, center - step / 2)
        hi = torch.where(q >= n, inf, center + step / 2)
        return lo, hi

    def log_mass(self, lo, hi, params):
        """ log P(lo < X < hi), lo and hi (batch, 1, time) """
        logits, locs, log_scales = self.split(params)
        lm = self._per_component(lambda f, lo, hi, m, s: f.log_mass(lo, hi, m, s),
                                 locs, log_scales, lo, hi)
        return torch.logsumexp(F.log_softmax(logits, dim=1) + lm, dim=1, keepdim=True)

    def log_prob(self, x, params):
        """ Log density (continuous), or log bin mass with discretize_bits """
        if self.discretize_bits is not None:
            return self.log_mass(*self.bin_edges(x), params)
        logits, locs, log_scales = self.split(params)
        lp = self._per_component(lambda f, x, m, s: f.log_prob(x, m, s),
                                 locs, log_scales, x)
        return torch.logsumexp(F.log_softmax(logits, dim=1) + lp, dim=1, keepdim=True)

    def cdf(self, x, params):
        """ Mixture CDF (of the continuous mixture, also when discretized) """
        logits, locs, log_scales = self.split(params)
        lc = self._per_component(lambda f, x, m, s: f.log_cdf(x, m, s),
                                 locs, log_scales, x)
        return torch.logsumexp(F.log_softmax(logits, dim=1) + lc, dim=1, keepdim=True).exp()

    # --- loss ---

    def nll(self, x, params):
        """ Negative log-likelihood (nats), shape (batch, 1, time).
            Continuous: of the density; discretized: of the bin mass """
        nll = -self.log_prob(x, params)
        logits, _, log_scales = self.split(params)
        # hinge on the entropy floor, per component
        self.batch_penalty = (self.log_scale_floor - log_scales).clamp(min=0.0).pow(2)
        self.batch_nll = nll
        self.batch_log_scale = log_scales.detach()
        self.batch_weights = F.softmax(logits.detach(), dim=1)
        return nll

    def forward(self, x, params):
        nll = self.nll(x, params)
        return (self.weight_nll * nll.mean()
                + self.weight_entropy_penalty * self.batch_penalty.mean())

    def log_stats(self):
        """ Statistics of the last nll() call, for logging """
        stats = {'entropy_floor_penalty': self.batch_penalty.mean().item()}
        weights = self.batch_weights.mean(dim=(0, 2))
        min_log_scale = self.batch_log_scale.amin(dim=(0, 2))
        for k, name in enumerate(self.components):
            stats[f'mixture/{k}_{name}/weight'] = weights[k].item()
            stats[f'mixture/{k}_{name}/min_log_scale'] = min_log_scale[k].item()
        return stats

    # --- sampling ---

    def set_temperature(self, temp: float):
        self.temperature = temp

    def sample(self, params, use_extension=False, temperature=None):
        """ Sample signal values, shape (batch, 1, time)

        A component is drawn from the mixture weights (not tempered), then a
        value from that component with its scale multiplied by the
        temperature. Temperature 0 returns the location of the most probable
        component. With discretize_bits, samples are snapped to the bins in [-1, 1].
        """
        self._check_params(params)
        if temperature is None:
            temperature = self.temperature
        temperature = torch.as_tensor(temperature, dtype=params.dtype, device=params.device)
        temperature = temperature.expand(params.size(0), 1, params.size(2))

        if use_extension:
            # seed drawn from torch's RNG, so torch.manual_seed makes it reproducible
            seed = int(torch.randint(0, 2**62, (1,)).item())
            x, = ext.sample_mixture(
                params.transpose(1, 2).contiguous(),       # (B, T, P)
                temperature.transpose(1, 2).contiguous(),  # (B, T, 1)
                list(self.components), self.shared_loc,
                self.discretize_bits or 0, seed)
            return x.transpose(1, 2)  # (B, 1, T)

        logits, locs, log_scales = self.split(params)

        k = torch.distributions.Categorical(logits=logits.transpose(1, 2)).sample()
        k = torch.where(temperature[:, 0] <= 0.0, logits.argmax(dim=1), k).unsqueeze(1)
        loc = locs.gather(1, k)
        log_scale = log_scales.gather(1, k)
        # a standard draw from every family, then pick the component's family
        z = torch.cat([family.sample_standard(loc) for family, _ in self._groups], dim=1)
        z = z.gather(1, self._family_of_component[k])
        x = loc + temperature * torch.exp(log_scale) * z

        if self.discretize_bits is not None:
            n = 2 ** self.discretize_bits - 1
            x = torch.round((x.clamp(-1.0, 1.0) + 1.0) * (n / 2.0)) * (2.0 / n) - 1.0
        return x


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
