""" Location-scale families (glotnet.losses.location_scale) """
import math

import pytest
import torch
from torch.distributions import (AffineTransform, Laplace, Normal,
                                 SigmoidTransform, TransformedDistribution,
                                 Uniform)

from glotnet.losses.location_scale import (FAMILIES, get_family,
                                           uniform_bin_entropy)

NAMES = sorted(FAMILIES)


def reference(name, loc, scale):
    if name == 'gaussian':
        return Normal(loc, scale)
    if name == 'laplace':
        return Laplace(loc, scale)
    if name == 'logistic':
        base = Uniform(torch.zeros_like(loc), torch.ones_like(loc))
        return TransformedDistribution(
            base, [SigmoidTransform().inv, AffineTransform(loc, scale)])
    raise KeyError(name)


def reference_cdf(name, x, loc, scale):
    if name == 'logistic':
        return torch.sigmoid((x - loc) / scale)
    if name == 'gaussian':
        # Normal.cdf and torch.special.ndtr cancel in the left tail; erfc does not
        return 0.5 * torch.special.erfc(-(x - loc) / (scale * math.sqrt(2.0)))
    return reference(name, loc, scale).cdf(x)


def random_params(n=1000, dtype=torch.float64):
    g = torch.Generator().manual_seed(0)
    x = 2 * torch.randn(n, generator=g, dtype=dtype)
    loc = torch.randn(n, generator=g, dtype=dtype)
    log_scale = torch.empty(n, dtype=dtype).uniform_(-5, 1, generator=g)
    return x, loc, log_scale


@pytest.mark.parametrize("name", NAMES)
def test_log_prob_matches_torch(name):
    x, loc, log_scale = random_params()
    family = get_family(name)
    ref = reference(name, loc, log_scale.exp()).log_prob(x)
    assert torch.allclose(family.log_prob(x, loc, log_scale), ref, atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("name", NAMES)
def test_cdf_matches_torch(name):
    x, loc, log_scale = random_params()
    family = get_family(name)
    ref = reference_cdf(name, x, loc, log_scale.exp())
    assert torch.allclose(family.cdf(x, loc, log_scale), ref, atol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_log_cdf_far_tails(name):
    """ float32 log-CDF stays finite and accurate where the CDF underflows """
    family = get_family(name)
    z = torch.tensor([-200.0, -60.0, -30.0, -5.0, 0.0, 5.0, 30.0, 60.0])
    exact = {
        'gaussian': lambda z: torch.special.log_ndtr(z.double()),
        'laplace': lambda z: torch.where(z < 0, z.double() - math.log(2),
                                         torch.log1p(-0.5 * torch.exp(-z.double()))),
        'logistic': lambda z: torch.nn.functional.logsigmoid(z.double()),
    }[name](z)
    out = family.log_cdf_standard(z)
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()
    assert torch.allclose(out.double(), exact, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("bits", [8, 16])
def test_log_mass_partition_sums_to_one(name, bits):
    """ Masses of a discretization of the real line (open edge bins) sum to 1 """
    family = get_family(name)
    levels = torch.linspace(-1, 1, 2**bits)
    half = (levels[1] - levels[0]) / 2
    lo = torch.cat([torch.tensor([-math.inf]), levels[1:] - half])
    hi = torch.cat([levels[:-1] + half, torch.tensor([math.inf])])
    for loc, log_scale in [(0.0, -2.0), (0.3, -6.0), (-0.95, -12.0), (0.0, 1.0)]:
        lm = family.log_mass(lo, hi, torch.tensor(loc), torch.tensor(log_scale))
        assert torch.isfinite(lm).all()
        total = torch.logsumexp(lm.double(), dim=0)
        assert abs(total.item()) < 1e-4, (loc, log_scale, total)


@pytest.mark.parametrize("name", NAMES)
def test_log_mass_matches_float64_reference(name):
    """ float32 result vs float64 CDF difference, for wide and narrow bins
        (narrow bins switch to the midpoint rule) and in the tails """
    family = get_family(name)
    g = torch.Generator().manual_seed(1)
    n = 2000
    loc = torch.zeros(n, dtype=torch.float64)
    log_scale = torch.zeros(n, dtype=torch.float64)
    lo = torch.empty(n, dtype=torch.float64).uniform_(-12, 12, generator=g)
    width = torch.exp(torch.empty(n, dtype=torch.float64).uniform_(-12, 1, generator=g))
    # endpoints exactly representable in float32, so both see the same interval
    lo = lo.float().double()
    hi = (lo + width).float().double()
    # float64 reference with the same reflection, so it does not cancel either
    flip = (lo + hi) > 0
    a, b = torch.where(flip, -hi, lo), torch.where(flip, -lo, hi)
    ref = torch.log(reference_cdf(name, b, loc, 1.0) - reference_cdf(name, a, loc, 1.0))
    ok = torch.isfinite(ref)  # float64 itself underflows far out
    out = family.log_mass(lo.float(), hi.float(), loc.float(), log_scale.float())
    assert torch.isfinite(out).all()
    err = (out.double() - ref)[ok].abs()
    assert err.max() < 1e-3, err.max()


@pytest.mark.parametrize("name", NAMES)
def test_log_mass_gradients_finite_at_open_edges(name):
    family = get_family(name)
    lo = torch.tensor([-math.inf, -0.5, 0.2, 0.999, 0.0])
    hi = torch.tensor([-0.9, -0.5 + 1e-6, 0.3, math.inf, 1e-7])
    loc = torch.zeros(5, requires_grad=True)
    log_scale = torch.full((5,), -3.0, requires_grad=True)
    family.log_mass(lo, hi, loc, log_scale).sum().backward()
    assert torch.isfinite(loc.grad).all(), loc.grad
    assert torch.isfinite(log_scale.grad).all(), log_scale.grad


@pytest.mark.parametrize("name", NAMES)
def test_log_mass_gradcheck(name):
    family = get_family(name)
    lo = torch.tensor([-math.inf, -0.3, 0.1, 0.4], dtype=torch.float64)
    hi = torch.tensor([-0.2, -0.1, 0.1 + 1e-4, math.inf], dtype=torch.float64)
    loc = torch.tensor([0.0, 0.1, -0.2, 0.3], dtype=torch.float64, requires_grad=True)
    log_scale = torch.tensor([-1.0, -2.0, -0.5, -3.0], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda m, s: family.log_mass(lo, hi, m, s), (loc, log_scale))


@pytest.mark.parametrize("name", NAMES)
def test_sample_matches_cdf(name):
    """ Kolmogorov-Smirnov distance of the samples to the family's CDF """
    family = get_family(name)
    torch.manual_seed(0)
    n = 200_000
    loc, log_scale = torch.full((n,), 0.3), torch.full((n,), -1.0)
    x = family.sample(loc, log_scale).sort().values
    ecdf = torch.arange(1, n + 1) / n
    ks = (family.cdf(x, loc, log_scale) - ecdf).abs().max()
    assert ks < 2.0 / math.sqrt(n), ks  # 1.36 / sqrt(n) at p = 0.05


@pytest.mark.parametrize("name", NAMES)
def test_sample_temperature_and_seed(name):
    family = get_family(name)
    loc, log_scale = torch.randn(1000), torch.randn(1000)
    torch.manual_seed(3)
    x1 = family.sample(loc, log_scale, temperature=1.0)
    torch.manual_seed(3)
    x_half = family.sample(loc, log_scale, temperature=0.5)
    assert torch.allclose(x_half - loc, 0.5 * (x1 - loc), atol=1e-6)
    assert torch.equal(family.sample(loc, log_scale, temperature=0.0), loc)
    assert torch.isfinite(x1).all()


@pytest.mark.parametrize("name", NAMES)
def test_icdf_inverts_cdf(name):
    family = get_family(name)
    u = torch.linspace(1e-6, 1 - 1e-6, 1001, dtype=torch.float64)
    z = family.icdf_standard(u)
    assert torch.allclose(family.log_cdf_standard(z).exp(), u, atol=1e-12)


@pytest.mark.parametrize("name", NAMES)
def test_entropy(name):
    family = get_family(name)
    log_scale = torch.tensor([-9.0, -2.0, 0.5], dtype=torch.float64)
    if name == 'logistic':
        # Monte Carlo -E[log p]
        torch.manual_seed(0)
        z = family.sample(torch.zeros(400_000, dtype=torch.float64),
                          torch.zeros(400_000, dtype=torch.float64))
        mc = -family.log_pdf_standard(z).mean()
        assert abs(mc - family.standard_entropy) < 0.01
        ref = family.standard_entropy + log_scale
    else:
        ref = reference(name, torch.zeros(3, dtype=torch.float64), log_scale.exp()).entropy()
    assert torch.allclose(family.entropy(log_scale), ref)
    assert torch.allclose(family.log_scale_from_entropy(family.entropy(log_scale)), log_scale)


def test_uniform_bin_entropy():
    # 16-bit signed integers on [-1, 1): bin width 2**-15
    assert uniform_bin_entropy(16) == pytest.approx(-15 * math.log(2))
    assert uniform_bin_entropy(8) == pytest.approx(math.log(2 / 256))
    # a Gaussian at the 16-bit floor has sigma ~ 1/4 of a bin
    floor = get_family('gaussian').log_scale_from_entropy(uniform_bin_entropy(16))
    assert math.exp(floor) == pytest.approx(2**-15 / math.sqrt(2 * math.pi * math.e))


def test_unknown_family():
    with pytest.raises(ValueError, match="Unknown"):
        get_family('cauchy')
