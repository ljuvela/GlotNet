""" Location-scale families: Gaussian, Laplace, logistic

    Building blocks for mixture distributions. Every function takes the
    location and the log scale as tensors that broadcast against x; nothing
    is stored, so the families are plain stateless objects.

    Each family defines its standard (loc 0, scale 1) log-density, log-CDF
    and inverse-CDF sampler; the base class adds the affine transform, a
    numerically stable log probability mass of an interval (for discretized
    likelihoods) and the entropy <-> log scale conversion (for entropy floors).
"""
import math

import torch
import torch.nn.functional as F


def uniform_bin_entropy(num_bits: int) -> float:
    """ Differential entropy (nats) of a uniform density over one bin of a
        num_bits signed integer signal on [-1, 1): log(2 / 2**num_bits).

        A continuous model whose entropy drops below this value is more
        certain than the quantization of the data allows, so it serves as
        an entropy floor with clear units (e.g. num_bits=16 for 16-bit audio).
    """
    return math.log(2.0) - num_bits * math.log(2.0)


def _log1mexp(d: torch.Tensor) -> torch.Tensor:
    """ log(1 - exp(d)) for d < 0 (Maechler 2012) """
    return torch.where(d > -math.log(2.0),
                       torch.log(-torch.expm1(d)),
                       torch.log1p(-torch.exp(d)))


class LocationScaleFamily:
    """ Base class: x = loc + scale * z, z from the standard distribution """

    name: str = None
    # differential entropy of the standard distribution (nats)
    standard_entropy: float = None
    # midpoint rule for the interval mass when (hi - lo) / scale is below this
    midpoint_width: float = 1e-2

    # --- standard distribution, implemented by the families ---

    def log_pdf_standard(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def log_cdf_standard(self, z: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def icdf_standard(self, u: torch.Tensor) -> torch.Tensor:
        """ Inverse CDF for u in (0, 1) """
        raise NotImplementedError

    def sample_standard(self, like: torch.Tensor) -> torch.Tensor:
        """ Standard samples shaped like `like`, from torch's RNG """
        # open interval (0, 1): torch.rand can return exactly 0
        info = torch.finfo(like.dtype)
        u = torch.rand_like(like).clamp(min=info.tiny, max=1.0 - info.eps)
        return self.icdf_standard(u)

    # --- location-scale transform ---

    @staticmethod
    def standardize(x, loc, log_scale):
        return (x - loc) * torch.exp(-log_scale)

    def log_prob(self, x, loc, log_scale):
        """ Log density (nats) """
        return self.log_pdf_standard(self.standardize(x, loc, log_scale)) - log_scale

    def log_cdf(self, x, loc, log_scale):
        return self.log_cdf_standard(self.standardize(x, loc, log_scale))

    def cdf(self, x, loc, log_scale):
        return torch.exp(self.log_cdf(x, loc, log_scale))

    def log_mass(self, lo, hi, loc, log_scale):
        """ log P(lo < X < hi), stable in the tails and for narrow intervals

        lo may be -inf or hi +inf (the edge bins of a discretized signal),
        but not both. Interior intervals right of the location are reflected
        to the left, where log-CDF differences do not cancel (the families
        are symmetric). Intervals narrower than `midpoint_width` scale units
        use the midpoint rule density * width, which is more accurate there
        in float32. Infinite edges are replaced by finite stand-ins before
        any arithmetic, so the unselected branches have finite gradients.
        """
        lo, hi, loc, log_scale = torch.broadcast_tensors(lo, hi, loc, log_scale)
        lo_open = torch.isneginf(lo)
        hi_open = torch.isposinf(hi)
        zero = torch.zeros_like(lo)
        inv_scale = torch.exp(-log_scale)
        z_lo = (torch.where(lo_open, zero, lo) - loc) * inv_scale
        z_hi = (torch.where(hi_open, zero, hi) - loc) * inv_scale

        # edge bins: log F(z_hi) or log(1 - F(z_lo)) = log F(-z_lo)
        edge = torch.where(lo_open, self.log_cdf_standard(z_hi),
                           self.log_cdf_standard(-z_lo))

        # interior, exact: log(F(b) - F(a)) with the interval left of zero
        flip = (z_lo + z_hi) > 0
        a = torch.where(flip, -z_hi, z_lo)
        b = torch.where(flip, -z_lo, z_hi)
        log_fb = self.log_cdf_standard(b)
        d = self.log_cdf_standard(a) - log_fb
        # the clamp keeps unselected (empty or reversed) intervals finite
        exact = log_fb + _log1mexp(d.clamp(max=-1e-30))

        # interior, midpoint rule
        width = z_hi - z_lo
        midpoint = self.log_pdf_standard(0.5 * (z_lo + z_hi)) \
            + torch.log(width.clamp(min=1e-30))

        interior = torch.where(width < self.midpoint_width, midpoint, exact)
        return torch.where(lo_open | hi_open, edge, interior)

    def sample(self, loc, log_scale, temperature=1.0):
        """ loc + temperature * scale * z; temperature 0 returns loc """
        loc, log_scale = torch.broadcast_tensors(loc, log_scale)
        z = self.sample_standard(loc)
        return loc + temperature * torch.exp(log_scale) * z

    # --- entropy ---

    def entropy(self, log_scale):
        """ Differential entropy (nats) """
        return self.standard_entropy + log_scale

    def log_scale_from_entropy(self, entropy):
        """ Inverse of entropy(): the log scale with the given entropy """
        return entropy - self.standard_entropy


class Gaussian(LocationScaleFamily):
    """ Normal distribution, scale = standard deviation """

    name = 'gaussian'
    standard_entropy = 0.5 * math.log(2.0 * math.pi * math.e)
    _half_log_2pi = 0.5 * math.log(2.0 * math.pi)

    def log_pdf_standard(self, z):
        return -0.5 * z * z - self._half_log_2pi

    def log_cdf_standard(self, z):
        return torch.special.log_ndtr(z)

    def icdf_standard(self, u):
        return torch.special.ndtri(u)

    def sample_standard(self, like):
        return torch.randn_like(like)


class Laplace(LocationScaleFamily):
    """ Laplace distribution, scale b: density exp(-|x - loc| / b) / (2 b) """

    name = 'laplace'
    standard_entropy = 1.0 + math.log(2.0)
    _log2 = math.log(2.0)

    def log_pdf_standard(self, z):
        return -z.abs() - self._log2

    def log_cdf_standard(self, z):
        # F(z) = exp(z) / 2 for z < 0, 1 - exp(-z) / 2 otherwise;
        # clamped arguments keep both branches finite (no NaN gradients)
        left = z.clamp(max=0.0) - self._log2
        right = torch.log1p(-0.5 * torch.exp(-z.clamp(min=0.0)))
        return torch.where(z < 0, left, right)

    def icdf_standard(self, u):
        # z = -sign(u - 1/2) * log(1 - 2 |u - 1/2|)
        v = u - 0.5
        return -torch.sign(v) * torch.log1p(-2.0 * v.abs())


class Logistic(LocationScaleFamily):
    """ Logistic distribution, scale s: CDF sigmoid((x - loc) / s) """

    name = 'logistic'
    standard_entropy = 2.0

    def log_pdf_standard(self, z):
        # log sigmoid(z) + log sigmoid(-z)
        return -z - 2.0 * F.softplus(-z)

    def log_cdf_standard(self, z):
        return F.logsigmoid(z)

    def icdf_standard(self, u):
        return torch.log(u) - torch.log1p(-u)


FAMILIES = {cls.name: cls() for cls in (Gaussian, Laplace, Logistic)}


def get_family(name: str) -> LocationScaleFamily:
    family = FAMILIES.get(name)
    if family is None:
        raise ValueError(f"Unknown location-scale family '{name}' "
                         f"(expected one of {sorted(FAMILIES)})")
    return family
