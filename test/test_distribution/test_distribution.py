import torch

from torch.distributions.normal import Normal as NormalRef
from glotnet.losses.distributions import GaussianDensity

def power_spectral_density():
    """ PSD for testing sampling """
    pass

def test_gaussian():

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

if __name__ == "__main__":

    test_gaussian()