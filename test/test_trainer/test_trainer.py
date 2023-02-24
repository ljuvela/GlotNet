import os
import tempfile
import pytest
import torch
import torchaudio
from torch.utils.data import TensorDataset
from glotnet.data.audio_dataset import AudioDataset

from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.config import Config


# pytest fixture for temporary directory
@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir


def test_trainer(tempdir):

    batch_size = 1
    timesteps = 100
    channels = 1

    # data
    f0 = 10
    fs = 100
    dur = int(1.0 * timesteps / fs) 
    t = torch.linspace(0, dur * fs, dur * fs) / fs
    x = torch.sin(2 * torch.pi * f0 * t)

    # data shape is ((batch_size, channels, timesteps)
    x = x.unsqueeze(0).unsqueeze(0)

    dataset = TensorDataset(x)
    config = Config(batch_size=batch_size,
                    learning_rate=1e-5,
                    dataset_compute_mel=False,
                    log_dir=tempdir)
    
    trainer = TrainerWaveNet(config=config, dataset=dataset)

    trainer.fit(num_iters=1)
    loss_1 = trainer.batch_loss
    trainer.fit(num_iters=1)
    loss_2 = trainer.batch_loss

    assert loss_2 < loss_1, \
        "Training must decrease loss function value"


def test_trainer_conditional(tempdir):

    config = Config(batch_size=4, learning_rate=1e-5, pre_emphasis=0.85)

    # data
    f0 = 200
    fs =  config.sample_rate
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    config.log_dir = tempdir
    config.dataset_audio_dir = tempdir
    torchaudio.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerWaveNet(config=config)

    trainer.fit(num_iters=1)
    loss_1 = trainer.batch_loss
    trainer.fit(num_iters=1)
    loss_2 = trainer.batch_loss

    assert loss_2 < loss_1, \
        "Training must decrease loss function value"

def test_trainer_generate(tempdir):

    config = Config(
        batch_size=4, 
        learning_rate=1e-5,
        pre_emphasis=0.85,
        dilations=[1],
        residual_channels=4,
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
    config.dataset_audio_dir = tempdir
    torchaudio.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerWaveNet(config=config)
    x = trainer.generate()
    
    assert x.shape == (1, 1, config.segment_len), \
        f"Generated audio expected to have shape (1, 1, {config.segment_len}), got {x.shape}"


def test_resume_training(tempdir):
    
    config = Config(batch_size=4, learning_rate=1e-5)
    config.log_dir = tempdir

    # data
    seg_len = config.batch_size * config.segment_len
    x = torch.randn(1, seg_len)

    config.dataset_audio_dir = tempdir
    torchaudio.save(os.path.join(tempdir, f"data.wav"),
                    x, sample_rate=config.sample_rate)

    trainer1 = TrainerWaveNet(config=config)
    trainer1.fit(num_iters=1)

    model_pt = os.path.join(tempdir, 'model.pt')
    optim_pt = os.path.join(tempdir, "optim.pt")
    trainer1.save(model_path=model_pt, optim_path=optim_pt)

    trainer2 = TrainerWaveNet(config=config)
    trainer2.load(model_path=model_pt, optim_path=optim_pt)

    for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
        assert torch.allclose(p1, p2)



def test_trainer_condnet(tempdir):

    config = Config(batch_size=4,
                    learning_rate=1e-5,
                    pre_emphasis=0.85,
                    input_channels=1,
                    use_condnet=True)

    # data
    f0 = 200
    fs =  config.sample_rate
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    config.log_dir = tempdir
    config.dataset_audio_dir = tempdir
    torchaudio.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerWaveNet(config=config)

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
                    use_condnet=True)

    # data
    f0 = 200
    fs =  config.sample_rate
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    config.log_dir = tempdir
    config.dataset_audio_dir = tempdir
    torchaudio.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = TrainerWaveNet(config=config)

    x = trainer.generate()

