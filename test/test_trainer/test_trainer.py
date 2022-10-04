import os
import tempfile
import torch
import torchaudio
from torch.utils.data import TensorDataset, DataLoader
from glotnet.data.audio_dataset import AudioDataset

from glotnet.trainer.trainer import Trainer
from glotnet.config import Config

def test_trainer():

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

def test_logging():

    batch_size = 4
    timesteps = 100
    channels = 1
    num_examples = 32

    x = torch.randn(num_examples, channels, timesteps)

    with tempfile.TemporaryDirectory() as dir:

        dataset = TensorDataset(x)
        config = Config(batch_size=batch_size)
        config.log_dir = dir
        criterion = Trainer.create_criterion(config)
        model = Trainer.create_model(config, criterion)


def test_resume_training():
    pass


def test_trainer_conditional():

   
    config = Config(batch_size=4, learning_rate=1e-5)

    # data
    f0 = 200
    fs =  config.sample_rate
    seg_len = config.batch_size * config.segment_len
    t = torch.linspace(0, seg_len, seg_len) / fs
    x = torch.sin(2 * torch.pi * f0 * t)
    x = x.unsqueeze(0)

    with tempfile.TemporaryDirectory() as dir:
        torchaudio.save(os.path.join(dir, f"sine.wav"),
                        x, sample_rate=config.sample_rate)

        config.cond_channels = config.n_mels
        dataset = AudioDataset(
            config=config,
            audio_dir=dir,
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



if __name__ == "__main__":

    print("Testing unconditional training")
    test_trainer()
    print("-- OK!")

    print("Testing conditional training")
    test_trainer_conditional()
    print("-- OK!")