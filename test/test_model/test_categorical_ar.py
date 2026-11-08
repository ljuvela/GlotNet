""" Autoregressive models with the mu-law categorical distribution """
import pytest
import torch

from glotnet.losses.distributions import CategoricalDensity, GaussianDensity
from glotnet.model.autoregressive.wavenet import WaveNetAR
from glotnet.model.autoregressive.glotnet import GlotNetAR
from glotnet.sigproc import mulaw


def sharpen_logits(model, gain=10.0):
    """ In an untrained model the argmax is set by the output bias; zero it
        and scale the final weights so the argmax follows the hidden state
        and the greedy path is not constant """
    with torch.no_grad():
        model.output2.conv.weight.mul_(gain)
        model.output2.conv.bias.zero_()
    return model


def make_wavenet_ar(dist, cond_channels=None):
    return WaveNetAR(input_channels=1, output_channels=dist.params_dim,
                     residual_channels=4, skip_channels=4,
                     kernel_size=3, dilations=[1, 2],
                     cond_channels=cond_channels, distribution=dist)


@pytest.mark.parametrize("cond_channels", [None, 2])
def test_wavenet_ar_categorical_greedy_matches_reference(cond_channels):
    """ C++ inference == Python reference loop at zero temperature """
    torch.manual_seed(0)
    model = sharpen_logits(make_wavenet_ar(CategoricalDensity(num_bits=8), cond_channels))
    timesteps = 90
    # nonzero excitation so the fed-back samples (and the argmax) vary over time
    x = 0.3 * torch.randn(1, 1, timesteps)
    c = None if cond_channels is None else 0.5 * torch.randn(1, cond_channels, timesteps)
    temp = torch.zeros(1, 1, timesteps)

    y_ref = model.forward(input=x, cond_input=c, temperature=temp)
    y_ext = model.inference(input=x, cond_input=c, temperature=temp)
    assert torch.allclose(y_ref, y_ext, atol=1e-6), (y_ref, y_ext)
    sampled = y_ref - x  # categorical part of the output
    assert sampled.unique().numel() > 5, "greedy path should not be constant"



def bump_logits(model, gain=3.0, center=160.0, width=8.0):
    """ Output bias = Gaussian bump over the classes, so the sampled class
        index has std ~ width * sqrt(T) and the temperature is measurable;
        scaled weights keep the dependence on the fed-back samples """
    k = torch.arange(model.output_channels, dtype=torch.float32)
    with torch.no_grad():
        model.output2.conv.weight.mul_(gain)
        model.output2.conv.bias.copy_(-(k - center) ** 2 / (2 * width ** 2))
    return model


def test_wavenet_ar_categorical_sampling_statistics_match_reference():
    """ T > 0: C++ and Python samplers use different RNGs, so compare the
        per-timestep mean and std of the sampled class index.
        (A C++ temperature of 1.0 instead of 0.7 gives std ratios ~1.2) """
    torch.manual_seed(0)
    model = bump_logits(make_wavenet_ar(CategoricalDensity(num_bits=8)))
    batch, timesteps, temperature = 4000, 6, 0.7
    x = torch.zeros(batch, 1, timesteps)
    temp = torch.full((batch, 1, timesteps), temperature)

    q_ref = mulaw.encode(model.forward(input=x, temperature=temp)).float()[:, 0, :]
    q_ext = mulaw.encode(model.inference(input=x, temperature=temp)).float()[:, 0, :]

    se = (q_ref.var(dim=0) / batch + q_ext.var(dim=0) / batch).sqrt()
    mean_diff = (q_ref.mean(dim=0) - q_ext.mean(dim=0)).abs()
    assert torch.all(mean_diff < 5 * se), (mean_diff, se)
    std_ratio = q_ext.std(dim=0) / q_ref.std(dim=0)
    assert torch.all((std_ratio - 1).abs() < 0.08), std_ratio


def test_wavenet_ar_categorical_sampling():
    model = make_wavenet_ar(CategoricalDensity(num_bits=8))
    x = torch.zeros(2, 1, 500)

    torch.manual_seed(1)
    y1 = model.inference(input=x)
    torch.manual_seed(1)
    y2 = model.inference(input=x)
    y3 = model.inference(input=x)

    assert y1.shape == x.shape
    assert torch.equal(y1, y2), "torch.manual_seed must make inference reproducible"
    assert not torch.equal(y1, y3)
    # zero excitation input: every output sample is a mu-law level
    assert y1.abs().max() <= 1.0
    assert torch.equal(mulaw.decode(mulaw.encode(y1)), y1)


def test_inference_rejects_mismatched_distribution():
    # teacher-forced use with other output sizes stays possible ...
    model = WaveNetAR(input_channels=1, output_channels=1,
                      residual_channels=4, skip_channels=4, kernel_size=3,
                      dilations=[1], distribution=GaussianDensity())
    # ... but C++ inference needs matching parameters
    with pytest.raises(ValueError):
        model.inference(input=torch.zeros(1, 1, 10))


def test_glotnet_ar_categorical_greedy_matches_reference():
    torch.manual_seed(0)
    order = 2
    dist = CategoricalDensity(num_bits=8)
    model = GlotNetAR(input_channels=3, output_channels=dist.params_dim,
                      residual_channels=4, skip_channels=4,
                      kernel_size=2, dilations=[1], distribution=dist,
                      hop_length=1, lpc_order=order)
    sharpen_logits(model)
    timesteps = 90
    x = 0.3 * torch.randn(1, 1, timesteps)
    a = torch.zeros(1, order + 1, timesteps)
    a[:, 0, :] = 1.0
    a[:, 1, :] = -0.5
    a[:, 2, :] = 0.2
    temp = torch.zeros(1, 1, timesteps)

    y_ref = model.forward(input=x, a=a, temperature=temp)
    y_ext = model.inference(input=x, a=a, temperature=temp)
    assert torch.allclose(y_ref, y_ext, atol=1e-5), (y_ref, y_ext)
    assert y_ref.unique().numel() > 5, "greedy path should not be constant"


def test_glotnet_ar_categorical_rejects_sample_after_filtering():
    dist = CategoricalDensity(num_bits=8)
    with pytest.raises(ValueError):
        GlotNetAR(input_channels=3, output_channels=dist.params_dim,
                  residual_channels=4, skip_channels=4,
                  kernel_size=2, dilations=[1], distribution=dist,
                  hop_length=1, lpc_order=2, sample_after_filtering=True)
