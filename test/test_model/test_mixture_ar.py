""" Autoregressive models with LocationScaleMixture: C++ inference vs the
    Python reference loop """
import pytest
import torch

from glotnet.losses.distributions import LocationScaleMixture
from glotnet.model.autoregressive.wavenet import WaveNetAR
from glotnet.model.autoregressive.glotnet import GlotNetAR

MIXTURES = {
    'gaussian+laplace': dict(components=('gaussian', 'laplace')),
    'shared_loc': dict(components=('gaussian', 'laplace'), shared_loc=True),
    'interleaved': dict(components=('laplace', 'gaussian', 'logistic', 'laplace')),
    'discretized': dict(components=('logistic',) * 3, discretize_bits=8),
}


def make_wavenet_ar(dist, cond_channels=None):
    return WaveNetAR(input_channels=1, output_channels=dist.params_dim,
                     residual_channels=4, skip_channels=4,
                     kernel_size=3, dilations=[1, 2],
                     cond_channels=cond_channels, distribution=dist)


def make_glotnet_ar(dist, sample_after_filtering, cond_channels=None, order=2):
    return GlotNetAR(input_channels=3, output_channels=dist.params_dim,
                     residual_channels=4, skip_channels=4,
                     kernel_size=2, dilations=[1, 2],
                     cond_channels=cond_channels, distribution=dist,
                     hop_length=1, lpc_order=order,
                     sample_after_filtering=sample_after_filtering)


def lpc_coefficients(batch, order, timesteps):
    a = torch.zeros(batch, order + 1, timesteps)
    a[:, 0, :] = 1.0
    a[:, 1, :] = -0.5
    a[:, 2, :] = 0.2
    return a


@pytest.mark.parametrize("cond_channels", [None, 2])
@pytest.mark.parametrize("mixture", sorted(MIXTURES))
def test_wavenet_ar_mixture_greedy_matches_reference(mixture, cond_channels):
    """ C++ inference == Python reference at zero temperature """
    torch.manual_seed(0)
    dist = LocationScaleMixture(**MIXTURES[mixture])
    model = make_wavenet_ar(dist, cond_channels)
    timesteps = 90
    x = 0.3 * torch.randn(1, 1, timesteps)
    c = None if cond_channels is None else 0.5 * torch.randn(1, cond_channels, timesteps)
    temp = torch.zeros(1, 1, timesteps)

    y_ref = model.forward(input=x, cond_input=c, temperature=temp)
    y_ext = model.inference(input=x, cond_input=c, temperature=temp)
    assert torch.allclose(y_ref, y_ext, atol=1e-5), (y_ref - y_ext).abs().max()
    assert (y_ref - x).unique().numel() > 5, "greedy path should not be constant"


@pytest.mark.parametrize("cond_channels", [None, 1])
@pytest.mark.parametrize("sample_after_filtering", [False, True])
@pytest.mark.parametrize("mixture", sorted(MIXTURES))
def test_glotnet_ar_mixture_greedy_matches_reference(mixture, sample_after_filtering,
                                                     cond_channels):
    """ C++ == Python at T = 0, batch 2, with and without sample_after_filtering
        (C++ conditional inference used to ignore sample_after_filtering) """
    torch.manual_seed(1)
    order, batch, timesteps = 2, 2, 60
    dist = LocationScaleMixture(**MIXTURES[mixture])
    model = make_glotnet_ar(dist, sample_after_filtering, cond_channels, order)
    x = 0.3 * torch.randn(batch, 1, timesteps)
    a = lpc_coefficients(batch, order, timesteps)
    c = None if cond_channels is None else 0.5 * torch.randn(batch, cond_channels, timesteps)
    temp = torch.zeros(batch, 1, timesteps)

    y_ref = model.forward(input=x, a=a, cond_input=c, temperature=temp)
    y_ext = model.inference(input=x, a=a, cond_input=c, temperature=temp)
    assert torch.allclose(y_ref, y_ext, atol=1e-5), (y_ref - y_ext).abs().max()
    # a snapped greedy path can settle on a grid value, but must not start there
    min_unique = 3 if 'discretize_bits' in MIXTURES[mixture] else 5
    assert y_ref.unique().numel() > min_unique, "greedy path should not be constant"


@pytest.mark.parametrize("cond_channels", [None, 1])
def test_glotnet_ar_discretized_sample_after_filtering_on_grid(cond_channels):
    """ Discretized + sample_after_filtering: the signal is on the grid
        (the bins follow x, not the excitation) """
    torch.manual_seed(2)
    bits, order, timesteps = 6, 2, 100  # the Python reference allows <= 100
    dist = LocationScaleMixture(('gaussian', 'laplace'), discretize_bits=bits)
    a = lpc_coefficients(1, order, timesteps)
    c = None if cond_channels is None else 0.5 * torch.randn(1, cond_channels, timesteps)
    n = 2 ** bits - 1
    for after in [True, False]:
        model = make_glotnet_ar(dist, after, cond_channels, order)
        for use_ext in [True, False]:
            fn = model.inference if use_ext else model.forward
            y = fn(input=torch.zeros(1, 1, timesteps), a=a, cond_input=c,
                   temperature=torch.ones(1, 1, timesteps))
            q = (y + 1) * n / 2
            off_grid = ((q - q.round()).abs() > 1e-4).sum().item()
            if after:
                assert off_grid == 0, (use_ext, off_grid)
            else:
                assert off_grid > timesteps // 2, (use_ext, off_grid)


def scale_dominated(model, dist, log_scale_range=(-1.5, -1.0)):
    """ Small location weights and the mixture's initial bias, so the sample
        spread comes from the component scales (and the temperature) """
    with torch.no_grad():
        model.output2.conv.weight.mul_(0.1)
        model.output2.conv.bias.copy_(dist.initial_bias(log_scale_range))
    return model


@pytest.mark.parametrize("mixture", ['gaussian+laplace', 'interleaved'])
def test_wavenet_ar_mixture_sampling_statistics_match_reference(mixture):
    """ T > 0: C++ and Python samplers use different RNGs, so compare the
        per-timestep mean and std over a batch.
        (A C++ temperature of 1.0 instead of 0.7 gives std ratios ~1.4.
        Sampling noise at batch 2000, worst over 10 seeds: mean 4.5 se,
        |std ratio - 1| 0.093; hence 6 se and 0.15) """
    torch.manual_seed(0)
    dist = LocationScaleMixture(**MIXTURES[mixture])
    model = scale_dominated(make_wavenet_ar(dist), dist)
    batch, timesteps, temperature = 2000, 6, 0.7
    x = torch.zeros(batch, 1, timesteps)
    temp = torch.full((batch, 1, timesteps), temperature)

    y_ref = model.forward(input=x, temperature=temp)[:, 0, :]
    y_ext = model.inference(input=x, temperature=temp)[:, 0, :]

    se = (y_ref.var(dim=0) / batch + y_ext.var(dim=0) / batch).sqrt()
    mean_diff = (y_ref.mean(dim=0) - y_ext.mean(dim=0)).abs()
    assert torch.all(mean_diff < 6 * se), (mean_diff, se)
    std_ratio = y_ext.std(dim=0) / y_ref.std(dim=0)
    assert torch.all((std_ratio - 1).abs() < 0.15), std_ratio


def test_glotnet_ar_mixture_sampling_statistics_match_reference():
    torch.manual_seed(0)
    order, batch, timesteps, temperature = 2, 2000, 6, 0.7
    dist = LocationScaleMixture(('gaussian', 'laplace'), shared_loc=True)
    model = scale_dominated(make_glotnet_ar(dist, True, order=order), dist)
    x = torch.zeros(batch, 1, timesteps)
    a = lpc_coefficients(batch, order, timesteps)
    temp = torch.full((batch, 1, timesteps), temperature)

    y_ref = model.forward(input=x, a=a, temperature=temp)[:, 0, :]
    y_ext = model.inference(input=x, a=a, temperature=temp)[:, 0, :]

    se = (y_ref.var(dim=0) / batch + y_ext.var(dim=0) / batch).sqrt()
    assert torch.all((y_ref.mean(dim=0) - y_ext.mean(dim=0)).abs() < 6 * se)
    std_ratio = y_ext.std(dim=0) / y_ref.std(dim=0)
    assert torch.all((std_ratio - 1).abs() < 0.15), std_ratio


def test_mixture_inference_seeded():
    dist = LocationScaleMixture(('gaussian', 'laplace'))
    model = make_wavenet_ar(dist)
    x = torch.zeros(2, 1, 300)
    torch.manual_seed(1)
    y1 = model.inference(input=x)
    torch.manual_seed(1)
    y2 = model.inference(input=x)
    assert torch.equal(y1, y2)
    assert not torch.equal(y1, model.inference(input=x))


def test_cpp_spec():
    assert LocationScaleMixture(('gaussian', 'laplace'), shared_loc=True,
                                discretize_bits=16).cpp_spec() == {
        'name': 'mixture', 'components': ['gaussian', 'laplace'],
        'shared_loc': True, 'discretize_bits': 16}
    assert LocationScaleMixture(('logistic',)).cpp_spec()['discretize_bits'] == 0
