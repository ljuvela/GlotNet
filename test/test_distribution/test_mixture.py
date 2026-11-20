""" LocationScaleMixture (Python reference implementation) """
import math

import pytest
import torch
from torch.distributions import Categorical, Laplace, MixtureSameFamily, Normal

from glotnet.losses.distributions import GaussianDensity, LocationScaleMixture
from glotnet.losses.location_scale import get_family, uniform_bin_entropy


def random_params(dist, batch=3, time=50, dtype=torch.float32, seed=0):
    g = torch.Generator().manual_seed(seed)
    params = torch.randn(batch, dist.params_dim, time, generator=g, dtype=dtype)
    K = dist.num_components
    n_loc = 1 if dist.shared_loc else K
    params[:, K:K + n_loc] *= 0.3                 # locations
    params[:, K + n_loc:] = params[:, K + n_loc:] * 0.5 - 2.5  # log scales
    x = 0.3 * torch.randn(batch, 1, time, generator=g, dtype=dtype)
    return x, params


def reference_log_prob(components, x, logits, locs, log_scales):
    """ float64 log sum_k w_k p_k(x), component by component """
    log_w = torch.log_softmax(logits.double(), dim=1)
    terms = []
    for k, name in enumerate(components):
        loc, scale = locs[:, k:k + 1].double(), log_scales[:, k:k + 1].double().exp()
        if name == 'gaussian':
            lp = Normal(loc, scale).log_prob(x.double())
        elif name == 'laplace':
            lp = Laplace(loc, scale).log_prob(x.double())
        else:
            z = (x.double() - loc) / scale
            lp = -z - 2 * torch.nn.functional.softplus(-z) - scale.log()
        terms.append(log_w[:, k:k + 1] + lp)
    return torch.logsumexp(torch.cat(terms, dim=1), dim=1, keepdim=True)


def test_params_dim():
    assert LocationScaleMixture(('gaussian', 'laplace')).params_dim == 6
    assert LocationScaleMixture(('gaussian', 'laplace'), shared_loc=True).params_dim == 5
    assert LocationScaleMixture(('logistic',) * 10).params_dim == 30


@pytest.mark.parametrize("family,ref", [('gaussian', Normal), ('laplace', Laplace)])
def test_nll_matches_mixture_same_family(family, ref):
    dist = LocationScaleMixture((family,) * 4)
    x, params = random_params(dist, dtype=torch.float64)
    logits, locs, log_scales = dist.split(params)
    mix = MixtureSameFamily(Categorical(logits=logits.permute(0, 2, 1)),
                            ref(locs.permute(0, 2, 1), log_scales.exp().permute(0, 2, 1)))
    expected = -mix.log_prob(x[:, 0]).unsqueeze(1)
    assert torch.allclose(dist.nll(x, params), expected, atol=1e-10)


@pytest.mark.parametrize("shared_loc", [False, True])
def test_nll_mixed_families(shared_loc):
    components = ('laplace', 'gaussian', 'logistic', 'gaussian', 'laplace')
    dist = LocationScaleMixture(components, shared_loc=shared_loc)
    x, params = random_params(dist)
    logits, locs, log_scales = dist.split(params)
    expected = -reference_log_prob(components, x, logits, locs, log_scales)
    assert torch.allclose(dist.nll(x, params).double(), expected, atol=1e-5)


def test_shared_loc_equals_repeated_locations():
    components = ('gaussian', 'laplace')
    shared = LocationScaleMixture(components, shared_loc=True)
    indep = LocationScaleMixture(components)
    x, params = random_params(shared)
    logits, loc, log_scales = params[:, :2], params[:, 2:3], params[:, 3:]
    params_indep = torch.cat([logits, loc, loc, log_scales], dim=1)
    assert torch.allclose(shared.nll(x, params), indep.nll(x, params_indep))


def test_single_gaussian_equals_gaussian_density():
    dist = LocationScaleMixture(('gaussian',))
    x, params = random_params(dist)
    gaussian = GaussianDensity()
    assert torch.allclose(dist.nll(x, params), gaussian.nll(x, params[:, 1:]), atol=1e-6)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shared_loc", [False, True])
def test_discretized_masses_sum_to_one(bits, shared_loc):
    dist = LocationScaleMixture(('gaussian', 'laplace', 'logistic'),
                                shared_loc=shared_loc, discretize_bits=bits)
    _, params = random_params(dist, batch=1, time=4)
    levels = torch.linspace(-1, 1, 2 ** bits).view(-1, 1, 1)
    lp = dist.log_prob(levels.expand(-1, 1, 4), params.expand(levels.size(0), -1, -1))
    total = torch.logsumexp(lp.double(), dim=0)
    assert torch.allclose(total, torch.zeros_like(total), atol=1e-4), total


def test_discretized_nll_is_bin_mass():
    bits = 8
    dist = LocationScaleMixture(('gaussian', 'laplace'), discretize_bits=bits)
    _, params = random_params(dist, batch=1, time=1)
    params = params.expand(-1, -1, 6)  # same distribution at every timestep
    step = 2 / (2 ** bits - 1)
    #             interior, same bin as interior, lower edge, beyond -1, upper edge, beyond +1
    x = torch.tensor([[[0.1, 0.1 + 0.2 * step, -1.0, -1.7, 1.0, 3.0]]])
    nll = dist.nll(x, params)
    assert nll[0, 0, 0] == nll[0, 0, 1], "same bin, same likelihood"
    lo, hi = dist.bin_edges(x)
    assert lo[0, 0, 2] == -math.inf and lo[0, 0, 3] == -math.inf
    assert hi[0, 0, 4] == math.inf and hi[0, 0, 5] == math.inf
    center = torch.tensor(round(1.1 / step) * step - 1)
    assert torch.allclose(lo[0, 0, 0], center - step / 2)
    assert torch.allclose(hi[0, 0, 0], center + step / 2)
    assert torch.equal(dist.bin_edges(x[..., 1:2])[0], lo[..., 0:1])
    expected = -dist.log_mass(lo, hi, params)
    assert torch.allclose(nll, expected)


def test_shift_location():
    for shared_loc in [False, True]:
        dist = LocationScaleMixture(('gaussian', 'laplace', 'logistic'), shared_loc=shared_loc)
        x, params = random_params(dist)
        offset = 0.5 * torch.randn(x.shape)
        shifted = dist.shift_location(params, offset)
        assert torch.allclose(dist.nll(x + offset, shifted), dist.nll(x, params), atol=1e-5)
        logits, _, log_scales = dist.split(params)
        logits2, _, log_scales2 = dist.split(shifted)
        assert torch.equal(logits, logits2) and torch.equal(log_scales, log_scales2)


@pytest.mark.parametrize("discretize_bits", [None, 6])
@pytest.mark.parametrize("shared_loc", [False, True])
def test_gradcheck(discretize_bits, shared_loc):
    dist = LocationScaleMixture(('gaussian', 'laplace', 'logistic'), shared_loc=shared_loc,
                                discretize_bits=discretize_bits)
    x, params = random_params(dist, batch=2, time=4, dtype=torch.float64)
    x[0, 0, 0] = 1.0   # upper edge bin when discretized
    x[1, 0, 1] = -2.0  # beyond the lower edge
    params.requires_grad_(True)
    assert torch.autograd.gradcheck(lambda p: dist(x, p), (params,))


def test_entropy_floor_hinge():
    bits = 16
    dist = LocationScaleMixture(('gaussian', 'laplace', 'logistic'), entropy_floor_bits=bits)
    # each family's log-scale floor has the entropy of one 16-bit bin
    for k, name in enumerate(dist.components):
        h = get_family(name).entropy(dist.log_scale_floor[0, k, 0].item())
        assert h == pytest.approx(uniform_bin_entropy(bits))
    x, params = random_params(dist, batch=1, time=1)
    params[:, 6:] = dist.log_scale_floor[0, :, 0].view(1, 3, 1) + 0.1
    dist(x, params)
    assert dist.batch_penalty.sum() == 0
    params[:, 7] -= 1.1  # laplace 1.0 below its floor
    dist(x, params)
    assert dist.batch_penalty.sum().item() == pytest.approx(1.0, abs=1e-4)
    assert dist.batch_penalty[0, 1, 0] > 0

    no_floor = LocationScaleMixture(('gaussian',), entropy_floor_bits=None)
    x, params = random_params(no_floor)
    params[:, 2] = -40.0
    no_floor(x, params)
    assert no_floor.batch_penalty.sum() == 0


def test_log_stats():
    dist = LocationScaleMixture(('gaussian', 'laplace'))
    x, params = random_params(dist)
    dist(x, params)
    stats = dist.log_stats()
    assert stats['mixture/0_gaussian/weight'] + stats['mixture/1_laplace/weight'] == pytest.approx(1.0)
    assert stats['mixture/1_laplace/min_log_scale'] == pytest.approx(params[:, 5].min().item())  # channels: logits 0-1, locs 2-3, log scales 4-5
    assert 'entropy_floor_penalty' in stats


def test_initial_bias():
    dist = LocationScaleMixture(('laplace', 'gaussian', 'laplace', 'gaussian', 'logistic'))
    bias = dist.initial_bias(log_scale_range=(-5.0, -2.0))
    assert bias.shape == (dist.params_dim,)
    assert torch.all(bias[:10] == 0), "zero logits and locations"
    _, _, log_scales = dist.split(bias.view(1, -1, 1))
    entropy = [get_family(c).entropy(log_scales[0, k, 0].item()) for k, c in enumerate(dist.components)]
    gaussian = get_family('gaussian')
    # two components per family: the same two entropies for both families
    assert entropy[1] == pytest.approx(gaussian.entropy(-5.0))
    assert entropy[3] == pytest.approx(gaussian.entropy(-2.0))
    assert entropy[0] == pytest.approx(entropy[1]) and entropy[2] == pytest.approx(entropy[3])
    # a single component sits in the middle
    assert entropy[4] == pytest.approx(gaussian.entropy(-3.5))
    shared = LocationScaleMixture(('gaussian', 'laplace'), shared_loc=True)
    assert shared.initial_bias().shape == (5,)


@pytest.mark.parametrize("components", [('gaussian', 'laplace', 'logistic'),
                                        ('laplace', 'gaussian', 'laplace')])
def test_sample_matches_mixture_cdf(components):
    """ Kolmogorov-Smirnov distance to the mixture CDF; the second case
        interleaves families, so each component must draw from its own """
    dist = LocationScaleMixture(components)
    n = 200_000
    #                  logits             locations         log scales
    p = torch.tensor([0.5, -0.3, 0.1,  -0.2, 0.0, 0.4,  -1.0, -2.5, -1.5]).view(1, -1, 1)
    params = p.expand(n, -1, -1)
    torch.manual_seed(0)
    x = dist.sample(params)[:, 0, 0].sort().values
    cdf = dist.cdf(x.view(-1, 1, 1), params)[:, 0, 0]
    ks = (cdf - torch.arange(1, n + 1) / n).abs().max()
    assert ks < 2.0 / math.sqrt(n), ks


def test_sample_temperature_and_seed():
    dist = LocationScaleMixture(('gaussian', 'laplace'), shared_loc=True)
    _, params = random_params(dist, batch=4, time=200)
    loc = params[:, 2:3]
    torch.manual_seed(5)
    x1 = dist.sample(params, temperature=1.0)
    torch.manual_seed(5)
    x_half = dist.sample(params, temperature=0.5)
    # same component and draw: the deviation from the shared location halves
    assert torch.allclose(x_half - loc, 0.5 * (x1 - loc), atol=1e-6)
    torch.manual_seed(5)
    assert torch.equal(dist.sample(params, temperature=1.0), x1)


def test_sample_zero_temperature_is_most_probable_location():
    dist = LocationScaleMixture(('gaussian', 'laplace', 'gaussian'))
    _, params = random_params(dist)
    logits, locs, _ = dist.split(params)
    expected = locs.gather(1, logits.argmax(dim=1, keepdim=True))
    assert torch.equal(dist.sample(params, temperature=0.0), expected)
    # per-timestep temperature: zeros only where requested
    temp = torch.ones(params.size(0), 1, params.size(2))
    temp[..., ::2] = 0.0
    x = dist.sample(params, temperature=temp)
    assert torch.equal(x[..., ::2], expected[..., ::2])
    assert not torch.equal(x[..., 1::2], expected[..., 1::2])


def test_discretized_samples_on_grid():
    bits = 6
    dist = LocationScaleMixture(('gaussian', 'logistic'), discretize_bits=bits)
    _, params = random_params(dist, batch=8, time=500)
    params[:, 2:4] *= 5  # locations out to about +-1.5: exercise the clipping
    x = dist.sample(params)
    n = 2 ** bits - 1
    assert x.abs().max() <= 1.0
    q = (x + 1) * n / 2
    assert torch.allclose(q, q.round(), atol=1e-4)


def test_fit_recovers_gaussian_laplace_mixture():
    """ The motivating case: Gaussian (broad) + Laplace (peaky) around a
        shared location. Maximum likelihood recovers the weights """
    torch.manual_seed(0)
    n = 20_000
    is_laplace = torch.rand(n) < 0.3
    x = torch.where(is_laplace, Laplace(0.0, 0.01).sample((n,)), Normal(0.0, 0.1).sample((n,)))
    x = x.view(1, 1, n)
    dist = LocationScaleMixture(('gaussian', 'laplace'), shared_loc=True)
    theta = dist.initial_bias().clone().requires_grad_(True)
    optim = torch.optim.Adam([theta], lr=0.05)
    for _ in range(600):
        optim.zero_grad()
        loss = dist(x, theta.view(1, -1, 1).expand(1, -1, n))
        loss.backward()
        optim.step()
    weights = torch.softmax(theta[:2].detach(), dim=0)
    assert weights[1].item() == pytest.approx(0.3, abs=0.03)
    assert theta[3].exp().item() == pytest.approx(0.1, rel=0.1)    # Gaussian std
    assert theta[4].exp().item() == pytest.approx(0.01, rel=0.15)  # Laplace scale


def test_errors():
    with pytest.raises(ValueError, match="Unknown"):
        LocationScaleMixture(('gaussian', 'cauchy'))
    with pytest.raises(ValueError, match="at least one"):
        LocationScaleMixture(())
    with pytest.raises(ValueError, match="discretize_bits"):
        LocationScaleMixture(('gaussian',), discretize_bits=0)
    dist = LocationScaleMixture(('gaussian', 'laplace'))
    with pytest.raises(ValueError, match="channels"):
        dist.nll(torch.zeros(1, 1, 4), torch.zeros(1, 5, 4))
    with pytest.raises(ValueError, match="channels"):
        dist.sample(torch.zeros(1, 5, 4), use_extension=True)


@pytest.mark.parametrize("discretize_bits", [None, 8])
def test_responsibilities(discretize_bits):
    """ Bayes rule: w_k p_k(x) / sum_j w_j p_j(x), in component order """
    components = ('laplace', 'gaussian', 'laplace')
    dist = LocationScaleMixture(components, discretize_bits=discretize_bits)
    x, params = random_params(dist, dtype=torch.float64)
    r = dist.responsibilities(x, params)
    assert r.shape == (x.size(0), 3, x.size(2))
    assert torch.allclose(r.sum(dim=1), torch.ones_like(r[:, 0]))
    # by hand, one single-component mixture per component
    logits, locs, log_scales = dist.split(params)
    w = torch.softmax(logits, dim=1)
    terms = []
    for k, name in enumerate(components):
        single = LocationScaleMixture((name,), discretize_bits=discretize_bits)
        p_k = torch.cat([torch.zeros_like(x), locs[:, k:k + 1], log_scales[:, k:k + 1]], dim=1)
        terms.append(w[:, k:k + 1] * single.log_prob(x, p_k).exp())
    expected = torch.cat(terms, dim=1)
    expected = expected / expected.sum(dim=1, keepdim=True)
    assert torch.allclose(r, expected, atol=1e-10)


# --- C++ sampler (use_extension=True) ---

EXT_CASES = [
    (('gaussian', 'laplace', 'logistic'), False, None),
    (('laplace', 'gaussian', 'laplace'), False, None),  # interleaved families
    (('gaussian', 'laplace'), True, None),               # shared location
]


@pytest.mark.parametrize("components,shared_loc,bits", EXT_CASES)
@pytest.mark.parametrize("temperature", [1.0, 0.5])
def test_extension_sample_matches_mixture_cdf(components, shared_loc, bits, temperature):
    """ Kolmogorov-Smirnov distance of C++ samples to the (tempered) mixture CDF """
    dist = LocationScaleMixture(components, shared_loc=shared_loc, discretize_bits=bits)
    n = 100_000
    _, p = random_params(dist, batch=1, time=1, seed=4)
    torch.manual_seed(0)
    x = dist.sample(p.expand(1, -1, n).contiguous(), use_extension=True,
                    temperature=temperature)[0, 0].sort().values
    # temperature T multiplies the scales: add log T to the log scales
    K = dist.num_components
    n_loc = 1 if shared_loc else K
    p_t = p.clone()
    p_t[:, K + n_loc:] += math.log(temperature)
    cdf = dist.cdf(x.view(-1, 1, 1), p_t.expand(n, -1, -1))[:, 0, 0]
    ks = (cdf - torch.arange(1, n + 1) / n).abs().max()
    assert ks < 2.0 / math.sqrt(n), ks


def test_extension_sample_greedy_matches_python():
    for shared_loc in [False, True]:
        dist = LocationScaleMixture(('gaussian', 'laplace', 'gaussian'), shared_loc=shared_loc)
        _, params = random_params(dist)
        assert torch.equal(dist.sample(params, use_extension=True, temperature=0.0),
                           dist.sample(params, temperature=0.0))


def test_extension_sample_discretized_matches_python_grid():
    dist = LocationScaleMixture(('gaussian', 'logistic'), discretize_bits=6)
    _, params = random_params(dist, batch=4, time=300)
    params[:, 2:4] *= 5  # locations beyond +-1: clipping
    x = dist.sample(params, use_extension=True)
    n = 2 ** 6 - 1
    assert x.abs().max() <= 1.0 and (x.abs() == 1.0).any()
    q = (x + 1) * n / 2
    assert torch.allclose(q, q.round(), atol=1e-4)
    # greedy: the same snapped value as the Python reference
    assert torch.allclose(dist.sample(params, use_extension=True, temperature=0.0),
                          dist.sample(params, temperature=0.0), atol=1e-6)


def test_extension_sample_seeded_and_per_timestep_temperature():
    dist = LocationScaleMixture(('gaussian', 'laplace'))
    _, params = random_params(dist, batch=2, time=100)
    torch.manual_seed(7)
    x1 = dist.sample(params, use_extension=True)
    torch.manual_seed(7)
    x2 = dist.sample(params, use_extension=True)
    assert torch.equal(x1, x2)
    temp = torch.ones(2, 1, 100)
    temp[..., ::2] = 0.0
    x = dist.sample(params, use_extension=True, temperature=temp)
    greedy = dist.sample(params, temperature=0.0)
    assert torch.equal(x[..., ::2], greedy[..., ::2])
    assert not torch.equal(x[..., 1::2], greedy[..., 1::2])
