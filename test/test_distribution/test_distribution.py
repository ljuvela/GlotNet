import pytest
import torch
from torch.distributions.normal import Normal as NormalRef
import torch.nn.functional as F
from glotnet.losses.distributions import GaussianDensity, CategoricalDensity
from glotnet.sigproc import mulaw

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


def test_categorical_params_dim():
    assert CategoricalDensity(num_bits=8).params_dim == 256
    assert CategoricalDensity(num_bits=10).params_dim == 1024


def test_categorical_nll():
    batch, num_samples = 3, 50
    dist = CategoricalDensity(num_bits=8)
    x = torch.rand(batch, 1, num_samples) * 2.0 - 1.0
    logits = torch.randn(batch, dist.params_dim, num_samples)

    nll = dist.nll(x, logits)
    assert nll.shape == (batch, 1, num_samples)

    # reference: log-softmax picked at the mu-law class of each sample
    target = mulaw.encode(x, 255)  # (batch, 1, time)
    log_p = F.log_softmax(logits, dim=1)
    nll_ref = -log_p.gather(1, target)
    assert torch.allclose(nll, nll_ref, atol=1e-6)

    loss = dist(x, logits)
    assert torch.isclose(loss, nll_ref.mean())


def test_categorical_wrong_params_dim():
    dist = CategoricalDensity(num_bits=8)
    with pytest.raises(ValueError):
        dist.sample(torch.randn(1, 2, 10))
    with pytest.raises(ValueError):
        dist.nll(torch.zeros(1, 1, 10), torch.randn(1, 2, 10))


def test_categorical_sample_zero_temperature_is_argmax():
    dist = CategoricalDensity(num_bits=8)
    logits = torch.randn(4, dist.params_dim, 20)
    x = dist.sample(logits, temperature=0.0)
    assert x.shape == (4, 1, 20)
    assert torch.equal(x, mulaw.decode(logits.argmax(dim=1, keepdim=True), 255))


def test_categorical_sample_values_on_grid():
    dist = CategoricalDensity(num_bits=8)
    x = dist.sample(torch.randn(2, dist.params_dim, 100))
    assert x.abs().max() <= 1.0
    # every sample is exactly a decoded quantization level
    assert torch.equal(mulaw.decode(mulaw.encode(x, 255), 255), x)


def test_categorical_sample_histogram():
    """ Empirical class frequencies match softmax(logits / T) """
    torch.manual_seed(0)
    num_draws, num_bits = 100000, 4
    dist = CategoricalDensity(num_bits=num_bits)
    logits = torch.randn(dist.params_dim)

    for temperature in [1.0, 0.5]:
        params = logits.view(1, -1, 1).expand(num_draws, -1, 1)
        x = dist.sample(params, temperature=temperature)
        q = mulaw.encode(x[:, 0, 0], dist.mu)
        freq = torch.bincount(q, minlength=dist.params_dim).float() / num_draws
        p = torch.softmax(logits / temperature, dim=0)
        # 5 standard deviations of a binomial proportion
        tol = 5.0 * (p * (1 - p) / num_draws).sqrt() + 1e-4
        assert torch.all((freq - p).abs() < tol), (freq, p)


def test_categorical_per_sample_temperature():
    """ Temperature tensor (batch, 1, time): zero entries are greedy """
    dist = CategoricalDensity(num_bits=8)
    logits = torch.randn(2, dist.params_dim, 30)
    temperature = torch.ones(2, 1, 30)
    temperature[:, :, ::2] = 0.0
    x = dist.sample(logits, temperature=temperature)
    x_greedy = mulaw.decode(logits.argmax(dim=1, keepdim=True), 255)
    assert torch.equal(x[:, :, ::2], x_greedy[:, :, ::2])


@pytest.mark.parametrize("temperature", [1.0, 0.5])
def test_categorical_sample_extension_histogram(temperature):
    """ C++ sampler: frequencies match softmax(logits / T) """
    torch.manual_seed(1)
    num_draws, num_bits = 100000, 4
    dist = CategoricalDensity(num_bits=num_bits)
    logits = torch.randn(dist.params_dim)
    params = logits.view(1, -1, 1).expand(1, -1, num_draws)  # draws along time
    x = dist.sample(params, use_extension=True, temperature=temperature)
    assert x.shape == (1, 1, num_draws)
    q = mulaw.encode(x[0, 0], dist.mu)
    freq = torch.bincount(q, minlength=dist.params_dim).float() / num_draws
    p = torch.softmax(logits / temperature, dim=0)
    tol = 5.0 * (p * (1 - p) / num_draws).sqrt() + 1e-4
    assert torch.all((freq - p).abs() < tol), (freq, p)


def test_categorical_sample_extension_seeded_and_greedy():
    dist = CategoricalDensity(num_bits=8)
    logits = torch.randn(3, dist.params_dim, 40)
    torch.manual_seed(3)
    a = dist.sample(logits, use_extension=True)
    torch.manual_seed(3)
    b = dist.sample(logits, use_extension=True)
    assert torch.equal(a, b)
    # zero temperature: identical to the Python path
    greedy_ext = dist.sample(logits, use_extension=True, temperature=0.0)
    greedy_py = dist.sample(logits, use_extension=False, temperature=0.0)
    assert torch.allclose(greedy_ext, greedy_py, atol=1e-6)


def test_ar_set_distribution_checks_output_channels():
    import glotnet.cpp_extensions as ext
    impl = ext.WaveNetAR(1, 2, 4, 4, 0, 2, "gated", [1, 2])
    impl.set_distribution("gaussian")
    with pytest.raises(ValueError):
        impl.set_distribution("categorical", 8)
    impl = ext.WaveNetAR(1, 256, 4, 4, 0, 2, "gated", [1, 2])
    impl.set_distribution("categorical", num_bits=8)
    impl.set_seed(0)
    glot = ext.GlotNetAR(3, 256, 4, 4, 0, 2, "gated", [1, 2], 10, True)
    with pytest.raises(ValueError):
        glot.set_distribution("categorical", 8)


if __name__ == "__main__":

    test_gaussian_nll()
    test_gaussian_sample()
    test_gaussian_sample_extension()