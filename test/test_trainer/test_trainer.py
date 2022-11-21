import os
import tempfile
import pytest
import torch
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
from glotnet.data.audio_dataset import AudioDataset

from glotnet.trainer.trainer import Trainer
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
    config = Config(batch_size=batch_size, learning_rate=1e-5)
    criterion = Trainer.create_criterion(config)
    model = Trainer.create_model(config, criterion)

    config.log_dir = tempdir
    trainer = Trainer(model=model,
                      criterion=criterion,
                      dataset=dataset,
                      config=config)

    trainer.fit(num_iters=1)
    loss_1 = trainer.batch_loss
    trainer.fit(num_iters=1)
    loss_2 = trainer.batch_loss

    assert loss_2 < loss_1, \
        "Training must decrease loss function value"


def test_trainer_conditional(tempdir):

   
    config = Config(batch_size=4, learning_rate=1e-5)

    # data
    f0 = 200
    fs =  config.sample_rate
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    config.log_dir = tempdir
    torchaudio.save(os.path.join(tempdir, f"sine.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = config.n_mels
    dataset = AudioDataset(
        config=config,
        audio_dir=tempdir,
        output_mel=True)
    criterion = Trainer.create_criterion(config)
    model = Trainer.create_model(config, criterion)

    trainer = Trainer(model=model,
                    criterion=criterion,
                    dataset=dataset,
                    config=config)

    trainer.fit(num_iters=1)
    loss_1 = trainer.batch_loss
    trainer.fit(num_iters=1)
    loss_2 = trainer.batch_loss

    assert loss_2 < loss_1, \
        "Training must decrease loss function value"


def test_resume_training(tempdir):
    
    config = Config(batch_size=4, learning_rate=1e-5)
    config.log_dir = tempdir

    # data
    seg_len = config.batch_size * config.segment_len
    x = torch.randn(1, seg_len)

    torchaudio.save(os.path.join(tempdir, f"data.wav"),
                    x, sample_rate=config.sample_rate)

    config.cond_channels = config.n_mels
    dataset = AudioDataset(
        config=config,
        audio_dir=tempdir,
        output_mel=True)
            
    criterion = Trainer.create_criterion(config)

    model1 = Trainer.create_model(config, criterion)
    trainer1 = Trainer(model=model1,
                        criterion=criterion,
                        dataset=dataset,
                        config=config)

    model_pt = os.path.join(tempdir, 'model.pt')
    optim_pt = os.path.join(tempdir, "optim.pt")

    trainer1.fit(num_iters=1)
    trainer1.save(model_path=model_pt, optim_path=optim_pt)

    model2 = Trainer.create_model(config, criterion)
    trainer2 = Trainer(model=model2,
                        criterion=criterion,
                        dataset=dataset,
                        config=config)

    trainer2.load(model_path=model_pt, optim_path=optim_pt)

    for p1, p2 in zip(trainer1.model.parameters(), trainer2.model.parameters()):
        assert torch.allclose(p1, p2)
