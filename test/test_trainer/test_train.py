
import os
import tempfile
import pytest
import torch
import torchaudio
from glotnet.config import Config
from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.trainer.trainer_glotnet import TrainerGlotNet

from glotnet.train import main as train_main
from glotnet.train import parse_args

# pytest fixture for temporary directory
@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir

@pytest.fixture
def config(tempdir):
    config = Config(use_condnet=True)

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

    return config


def test_train(tempdir, config):

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True
    config.max_iters = 1

    # save model to tempdir
    trainer = TrainerWaveNet(config=config)
    trainer.save(model_path=os.path.join(tempdir, "model.pt"))
    config.to_json(os.path.join(tempdir, "config.json"))

    args = parse_args(args=[
        '--mel_cond', 'True',
        '--config', os.path.join(tempdir, 'config.json'),
        '--device', 'cpu',
        ])
    train_main(args)


def test_train_glotnet(tempdir, config):

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True
    config.max_iters = 1
    config.input_channels = 3

    # save model to tempdir
    trainer = TrainerGlotNet(config=config)
    trainer.save(model_path=os.path.join(tempdir, "model.pt"))
    config.to_json(os.path.join(tempdir, "config.json"))

    args = parse_args(args=[
        '--mel_cond', 'True',
        '--config', os.path.join(tempdir, 'config.json'),
        '--device', 'cpu',
        ])
    train_main(args)



def test_train_with_validation(tempdir, config):

    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True
    config.max_iters = 1

    config.dataset_audio_dir_validation = tempdir
    config.dataset_filelist_validation = config.dataset_filelist_training

    # save model to tempdir
    trainer = TrainerWaveNet(config=config)
    trainer.save(model_path=os.path.join(tempdir, "model.pt"))
    config.to_json(os.path.join(tempdir, "config.json"))

    args = parse_args(args=[
        '--mel_cond', 'True',
        '--config', os.path.join(tempdir, 'config.json'),
        '--device', 'cpu',
        ])
    train_main(args)