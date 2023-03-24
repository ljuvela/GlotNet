
import os
import tempfile
import pytest
import torch
import torchaudio
from torch.utils.data import TensorDataset
from glotnet.data.audio_dataset import AudioDataset
import argparse
from glotnet.config import Config
from glotnet.trainer.trainer import Trainer as TrainerWaveNet


# pytest fixture for temporary directory
@pytest.fixture
def tempdir():
    with tempfile.TemporaryDirectory() as tempdir:
        yield tempdir


def test_copy_synthesis(tempdir):

    config = Config(use_condnet=True)

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

    # save model to tempdir
    trainer = TrainerWaveNet(config=config)
    trainer.save(model_path=os.path.join(tempdir, "model.pt"))
    config.to_json(os.path.join(tempdir, "config.json"))

    from glotnet.copy_synthesis import main as copy_synthesis
    from glotnet.copy_synthesis import parse_args
    args = parse_args(args=[
        '--model', os.path.join(tempdir, 'model.pt'),
        '--config', os.path.join(tempdir, 'config.json'),
        '--output_dir', tempdir,
        '--input_dir', tempdir
        ])
    copy_synthesis(args)