import os
import tempfile
import pytest
import torch
from glotnet.data import audio_io

from torch.utils.data import TensorDataset
from glotnet.data.audio_dataset import AudioDataset

from glotnet.trainer.trainer_glotnet import TrainerGlotNet
from glotnet.config import Config

# pytest fixture for temporary directory
@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir


def test_trainer_generate(tempdir):

    config = Config(
        batch_size=4, 
        learning_rate=1e-5,
        pre_emphasis=0.85,
        dilations=[1],
        residual_channels=4,
        input_channels=3,
        skip_channels=4)

    # data
    f0 = 200
    fs =  config.sample_rate
    config.segment_len = 1000
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    config.log_dir = tempdir
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ['sine.wav']
    audio_io.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerGlotNet(config=config)
    data = trainer.create_dataset_training()
    trainer.set_training_dataset(data)
    x = trainer.generate(data)
    
    assert x.shape == (1, 1, config.segment_len), \
        f"Generated audio expected to have shape (1, 1, {config.segment_len}), got {x.shape}"


def test_trainer_condnet(tempdir):

    config = Config(batch_size=4,
                    learning_rate=1e-5,
                    pre_emphasis=0.85,
                    input_channels=3,
                    use_condnet=True)

    # data
    f0 = 200
    fs =  config.sample_rate
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    config.log_dir = tempdir
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ['sine.wav']
    audio_io.save(os.path.join(tempdir, f'sine.wav'),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerGlotNet(config=config)
    trainer.fit(num_iters=1)
    loss_1 = trainer.batch_loss
    trainer.fit(num_iters=1)
    loss_2 = trainer.batch_loss

    assert loss_2 < loss_1, \
        "Training must decrease loss function value"

def test_trainer_generate_condnet(tempdir):

    config = Config(batch_size=4,
                    learning_rate=1e-5,
                    pre_emphasis=0.85,
                    input_channels=3,
                    use_condnet=True)

    # data
    f0 = 200
    fs =  config.sample_rate
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    config.log_dir = tempdir
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ['sine.wav']
    audio_io.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerGlotNet(config=config)
    data = trainer.create_dataset_training()
    x = trainer.generate(data)


def test_trainer_condnet_sample_after_filtering(tempdir):

    config = Config(batch_size=4,
                    learning_rate=1e-5,
                    pre_emphasis=0.85,
                    input_channels=3,
                    use_condnet=True,
                    glotnet_sample_after_filtering=True)

    # data
    f0 = 200
    fs =  config.sample_rate
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    config.log_dir = tempdir
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ['sine.wav']
    audio_io.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerGlotNet(config=config)

    trainer.fit(num_iters=1)
    loss_1 = trainer.batch_loss
    trainer.fit(num_iters=1)
    loss_2 = trainer.batch_loss

    assert loss_2 < loss_1, \
        "Training must decrease loss function value"

def test_trainer_generate_condnet_sample_after_filtering(tempdir):

    config = Config(batch_size=4,
                    learning_rate=1e-5,
                    pre_emphasis=0.85,
                    input_channels=3,
                    use_condnet=True,
                    glotnet_sample_after_filtering=True)

    # data
    f0 = 200
    fs =  config.sample_rate
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    config.log_dir = tempdir
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ["sine.wav"]
    audio_io.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerGlotNet(config=config)

    dataset = trainer.create_dataset_training()
    x = trainer.generate(dataset)



@pytest.mark.parametrize("phase", ["fit", "validate"])
def test_trainer_sample_after_filtering_shifts_mean_only(tempdir, phase):
    """ Regression: the LPC prediction shifts the mean, not log_s
        (training and inference must agree on the shifted distribution) """
    config = Config(batch_size=2,
                    learning_rate=1e-5,
                    pre_emphasis=0.85,
                    dilations=[1, 2],
                    residual_channels=4,
                    skip_channels=4,
                    input_channels=3,
                    glotnet_sample_after_filtering=True)
    config.segment_len = 1000
    seg_len = config.batch_size * config.segment_len
    t = torch.arange(seg_len) / config.sample_rate
    x = 0.5 * torch.sin(2 * torch.pi * 200 * t).unsqueeze(0)
    config.log_dir = tempdir
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ['sine.wav']
    config.dataset_filelist_validation = ['sine.wav']
    config.dataset_audio_dir_validation = tempdir
    audio_io.save(os.path.join(tempdir, "sine.wav"), x, sample_rate=config.sample_rate)
    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerGlotNet(config=config)
    captured = {}
    trainer.model.register_forward_hook(
        lambda module, args, output: captured.update(raw=output.detach(), input=args[0].detach()))
    trainer.criterion.register_forward_hook(
        lambda module, args, kwargs, output: captured.update(params=kwargs['params'].detach()),
        with_kwargs=True)

    if phase == "fit":
        trainer.fit(num_iters=1)
    else:
        trainer.validate()

    pad = config.padding
    raw = captured['raw'][..., pad:]
    params = captured['params']
    p_curr = captured['input'][:, 1:2, pad:]  # model input channels: (e_prev, p_curr, x_prev)
    assert p_curr.abs().max() > 0.1, "test needs a nonzero LPC prediction"
    assert torch.allclose(params[:, 0:1], raw[:, 0:1] + p_curr), "mean must shift by the prediction"
    assert torch.equal(params[:, 1:2], raw[:, 1:2]), "log_s must not shift"
