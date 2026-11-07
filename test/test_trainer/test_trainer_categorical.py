""" Trainers with distribution = 'categorical' """
import os
import tempfile
import pytest
import torch
from glotnet.data import audio_io

from glotnet.config import Config
from glotnet.losses.distributions import CategoricalDensity
from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.trainer.trainer_glotnet import TrainerGlotNet


@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir


def make_config(tempdir, **kwargs):
    config = Config(
        batch_size=4,
        learning_rate=1e-4,
        pre_emphasis=0.85,
        dilations=[1, 2],
        residual_channels=4,
        skip_channels=4,
        distribution='categorical',
        num_bits=8,
        **kwargs)
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


def test_trainer_categorical_fit_and_generate(tempdir):
    config = make_config(tempdir)
    trainer = TrainerWaveNet(config=config)
    assert isinstance(trainer.criterion, CategoricalDensity)
    assert trainer.model.output_channels == 256

    trainer.fit(num_iters=1)
    loss_1 = trainer.batch_loss
    trainer.fit(num_iters=1)
    loss_2 = trainer.batch_loss
    assert loss_2 < loss_1, "Training must decrease loss function value"

    assert isinstance(trainer.model_ar.distribution, CategoricalDensity)
    x = trainer.generate(dataset=trainer.create_dataset_training())
    assert x.shape == (1, 1, config.segment_len)
    assert torch.isfinite(x).all()


def test_trainer_glotnet_categorical_generate(tempdir):
    config = make_config(tempdir, input_channels=3)
    trainer = TrainerGlotNet(config=config)
    trainer.fit(num_iters=1)

    model_ar = trainer.model_ar
    assert model_ar is trainer.model_ar, "model_ar must be cached"
    assert isinstance(model_ar.distribution, CategoricalDensity)
    data = trainer.create_dataset_training()
    x = trainer.generate(data)
    assert x.shape == (1, 1, config.segment_len)
    assert torch.isfinite(x).all()


def test_trainer_glotnet_categorical_rejects_sample_after_filtering(tempdir):
    config = make_config(tempdir, input_channels=3,
                         glotnet_sample_after_filtering=True)
    with pytest.raises(ValueError):
        TrainerGlotNet(config=config)
