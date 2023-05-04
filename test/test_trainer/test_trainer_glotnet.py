import os
import tempfile
import pytest
import torch
import torchaudio

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
    torchaudio.save(os.path.join(tempdir, f"sine.wav"),
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
    torchaudio.save(os.path.join(tempdir, f'sine.wav'),
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
    torchaudio.save(os.path.join(tempdir, f"sine.wav"),
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
    torchaudio.save(os.path.join(tempdir, f"sine.wav"),
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
    torchaudio.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerGlotNet(config=config)

    dataset = trainer.create_dataset_training()
    x = trainer.generate(dataset)

