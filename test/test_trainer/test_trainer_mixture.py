""" Trainers with distribution = 'mixture' (tempdir fixture: conftest.py) """
import os

import pytest
import torch
from glotnet.data import audio_io

from glotnet.config import Config
from glotnet.losses.distributions import GaussianDensity, LocationScaleMixture
from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.trainer.trainer_glotnet import TrainerGlotNet


def make_config(tempdir, **kwargs):
    defaults = dict(
        batch_size=4,
        learning_rate=1e-4,
        pre_emphasis=0.85,
        dilations=[1, 2],
        residual_channels=4,
        skip_channels=4,
        distribution='mixture',
        mixture_components=('gaussian', 'laplace'))
    defaults.update(kwargs)
    config = Config(**defaults)
    config.segment_len = 1000
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / config.sample_rate
    x = 0.5 * torch.sin(2 * torch.pi * 200 * t).unsqueeze(0)

    config.log_dir = tempdir
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ["sine.wav"]
    audio_io.save(os.path.join(tempdir, "sine.wav"), x, sample_rate=config.sample_rate)
    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True
    return config


@pytest.mark.parametrize("settings", [
    dict(),
    dict(mixture_shared_loc=True),
    dict(mixture_components=('logistic',) * 3, discretize_bits=16),
])
def test_trainer_mixture_fit_and_generate(tempdir, settings):
    config = make_config(tempdir, **settings)
    trainer = TrainerWaveNet(config=config)
    dist = trainer.criterion
    assert isinstance(dist, LocationScaleMixture)
    assert dist.components == tuple(config.mixture_components)
    assert dist.shared_loc == config.mixture_shared_loc
    assert dist.discretize_bits == config.discretize_bits
    assert trainer.model.output_channels == dist.params_dim
    # grouped, entropy-matched initial scales on the output layer
    assert torch.equal(trainer.model.output2.conv.bias.detach(), dist.initial_bias())

    trainer.fit(num_iters=1)
    loss_1 = trainer.batch_loss
    trainer.fit(num_iters=1)
    loss_2 = trainer.batch_loss
    assert loss_2 < loss_1, "Training must decrease loss function value"
    # per-component statistics reach the logger
    stats = dist.log_stats()
    assert 'mixture/0_%s/weight' % dist.components[0] in stats

    assert isinstance(trainer.model_ar.distribution, LocationScaleMixture)
    x = trainer.generate(dataset=trainer.create_dataset_training())
    assert x.shape == (1, 1, config.segment_len)
    assert torch.isfinite(x).all()


@pytest.mark.parametrize("sample_after_filtering", [False, True])
@pytest.mark.parametrize("discretize_bits", [None, 16])
def test_trainer_glotnet_mixture(tempdir, sample_after_filtering, discretize_bits):
    """ Mixtures support sample_after_filtering (locations shift by the prediction) """
    config = make_config(tempdir, input_channels=3, discretize_bits=discretize_bits,
                         glotnet_sample_after_filtering=sample_after_filtering)
    trainer = TrainerGlotNet(config=config)
    trainer.fit(num_iters=1)
    assert torch.isfinite(trainer.batch_loss)
    x = trainer.generate(trainer.create_dataset_training())
    assert x.shape == (1, 1, config.segment_len)
    assert torch.isfinite(x).all()


def test_trainer_glotnet_mixture_shifts_locations_only(tempdir):
    """ Training with sample_after_filtering shifts the location channels by
        the LPC prediction; logits and log scales are untouched """
    config = make_config(tempdir, input_channels=3, glotnet_sample_after_filtering=True)
    trainer = TrainerGlotNet(config=config)
    captured = {}
    trainer.model.register_forward_hook(
        lambda module, args, output: captured.update(raw=output.detach(), input=args[0].detach()))
    trainer.criterion.register_forward_hook(
        lambda module, args, kwargs, output: captured.update(params=kwargs['params'].detach()),
        with_kwargs=True)
    trainer.fit(num_iters=1)
    pad = config.padding
    raw, params = captured['raw'][..., pad:], captured['params']
    p_curr = captured['input'][:, 1:2, pad:]
    assert p_curr.abs().max() > 0.1
    assert torch.allclose(params[:, 2:4], raw[:, 2:4] + p_curr)  # locations
    assert torch.equal(params[:, :2], raw[:, :2])                # logits
    assert torch.equal(params[:, 4:], raw[:, 4:])                # log scales


def test_checkpoint_round_trip_with_components(tempdir):
    config = make_config(tempdir, mixture_components=('laplace', 'gaussian', 'laplace'),
                         mixture_shared_loc=True, discretize_bits=16)
    trainer = TrainerWaveNet(config=config)
    model_pt = os.path.join(tempdir, "model.pt")
    trainer.save(model_path=model_pt)

    saved = TrainerWaveNet.config_from_checkpoint(model_pt)
    assert saved.distribution == 'mixture'
    assert tuple(saved.mixture_components) == ('laplace', 'gaussian', 'laplace')
    assert saved.mixture_shared_loc and saved.discretize_bits == 16
    trainer2 = TrainerWaveNet(config=saved)
    trainer2.load(model_pt)
    for p1, p2 in zip(trainer.model.parameters(), trainer2.model.parameters()):
        assert torch.equal(p1, p2)

    # a config read from JSON has lists where the checkpoint has tuples
    config_json = os.path.join(tempdir, "config.json")
    config.to_json(config_json)
    from_json = Config.from_json(config_json)
    assert isinstance(from_json.mixture_components, list)
    TrainerWaveNet(config=from_json).load(model_pt)

    # different components (same parameter count) are reported
    other = make_config(tempdir, mixture_components=('gaussian', 'gaussian', 'laplace'),
                        mixture_shared_loc=True, discretize_bits=16)
    with pytest.raises(ValueError, match="mixture_components"):
        TrainerWaveNet(config=other).load(model_pt)
    other = make_config(tempdir, mixture_components=('laplace', 'gaussian', 'laplace'),
                        mixture_shared_loc=True, discretize_bits=None)
    with pytest.raises(ValueError, match="discretize_bits"):
        TrainerWaveNet(config=other).load(model_pt)


def test_checkpoint_ignores_mixture_fields_for_gaussian(tempdir):
    config = make_config(tempdir, distribution='gaussian')
    trainer = TrainerWaveNet(config=config)
    model_pt = os.path.join(tempdir, "model.pt")
    trainer.save(model_path=model_pt)
    # unused mixture settings must not block loading a Gaussian model
    config2 = make_config(tempdir, distribution='gaussian',
                          mixture_components=('logistic',), discretize_bits=8)
    TrainerWaveNet(config=config2).load(model_pt)


def test_gaussian_log_stats(tempdir):
    trainer = TrainerWaveNet(config=make_config(tempdir, distribution='gaussian'))
    assert isinstance(trainer.criterion, GaussianDensity)
    trainer.fit(num_iters=1)
    stats = trainer.criterion.log_stats()
    assert set(stats) == {'entropy_floor_penalty', 'min_log_scale'}
