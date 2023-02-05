import torch
from torch.distributions.normal import Normal as NormalRef
from glotnet.losses.distributions import GaussianDensity

def power_spectral_density(x):
    """ PSD for testing sampling """
    X = torch.fft.rfft(x).abs().pow(2)
    return X.mean(dim=0).sqrt()

def test_gaussian_nll():

    batch = 2
    num_samples = 10
    channels = 1
    x = torch.randn(batch, channels, num_samples)
    mu = torch.randn(batch, channels, num_samples)
    log_sigma = torch.clamp(torch.randn(batch, channels, num_samples), min=-7.0)
    sigma = torch.exp(log_sigma)
    
    dist_ref = NormalRef(loc=mu, scale=sigma)
    dist = GaussianDensity()

    # negative log likelihood from reference
    nll_ref = -1.0 * dist_ref.log_prob(x)
    # negative log likelihood from custom implementation
    params = torch.cat([mu, log_sigma], dim=1)
    nll = dist.nll(x, params)

    assert torch.allclose(nll_ref, nll), \
        f"NLL must match \n ref: {nll_ref} \n custom: {nll}"

def test_gaussian_sample():

    batch = 2048
    num_samples = 10
    channels = 1
    mu = torch.randn(1, channels, num_samples)
    log_sigma = torch.clamp(torch.randn(1, channels, num_samples), min=-7.0)
    sigma = torch.exp(log_sigma)

    # repeated sampling for batch elements
    mu = mu.expand(batch, -1, -1)
    log_sigma = log_sigma.expand(batch, -1, -1)
    sigma = sigma.expand(batch, -1, -1)
    
    dist_ref = NormalRef(loc=mu, scale=sigma)
    dist = GaussianDensity()

    x_ref = dist_ref.sample()

    params = torch.cat([mu, log_sigma], dim=1)
    x = dist.sample(params)

    X_ref = power_spectral_density(x_ref)
    X = power_spectral_density(x)

    # loose tolerances when scoring sample similarity
    assert torch.allclose(X_ref, X, atol=1e-2, rtol=1e-1), \
        f"PSDs must match \n ref: {X_ref} \n custom: {X}"


def test_gaussian_sample_extension():

    batch = 2048
    num_samples = 10
    channels = 1
    mu = torch.randn(1, channels, num_samples)
    log_sigma = torch.clamp(torch.randn(1, channels, num_samples), min=-7.0)
    sigma = torch.exp(log_sigma)

    # repeated sampling for batch elements
    mu = mu.expand(batch, -1, -1)
    log_sigma = log_sigma.expand(batch, -1, -1)
    sigma = sigma.expand(batch, -1, -1)
    
    dist_ref = NormalRef(loc=mu, scale=sigma)
    dist = GaussianDensity()

    x_ref = dist_ref.sample()

    params = torch.cat([mu, log_sigma], dim=1)
    x = dist.sample(params, use_extension=True)

    X_ref = power_spectral_density(x_ref)
    X = power_spectral_density(x)

    print(X_ref)
    print(X)

    # loose tolerances when scoring sample similarity
    assert torch.allclose(X_ref, X, atol=1e-2, rtol=1e-1), \
        f"PSDs must match \n ref: {X_ref} \n custom: {X}"
