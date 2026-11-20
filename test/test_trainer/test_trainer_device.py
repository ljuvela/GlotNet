""" Training on an accelerator, generation through the C++ (CPU) inference.
    Skipped without CUDA or MPS. (tempdir fixture: conftest.py) """
import os

import pytest
import torch
from glotnet.data import audio_io

from glotnet.config import Config
from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.trainer.trainer_glotnet import TrainerGlotNet

if torch.cuda.is_available():
    DEVICE = 'cuda'
elif torch.backends.mps.is_available():
    DEVICE = 'mps'
else:
    DEVICE = None
pytestmark = pytest.mark.skipif(DEVICE is None, reason="no CUDA or MPS device")


@pytest.mark.parametrize("use_condnet", [False, True])
@pytest.mark.parametrize("trainer_class,input_channels", [(TrainerWaveNet, 1), (TrainerGlotNet, 3)])
def test_generate_after_training_on_device(tempdir, trainer_class, input_channels, use_condnet):
    """ Regression: generate() mixed device tensors (pre-emphasis and LPC
        filters on the device, inference signals on the CPU) """
    config = Config(batch_size=2, learning_rate=1e-4, pre_emphasis=0.85,
                    dilations=[1, 2], residual_channels=4, skip_channels=4,
                    input_channels=input_channels, use_condnet=use_condnet,
                    distribution='mixture', mixture_components=('gaussian', 'laplace'))
    config.segment_len = 1000
    t = torch.arange(config.batch_size * config.segment_len) / config.sample_rate
    x = 0.5 * torch.sin(2 * torch.pi * 200 * t).unsqueeze(0)
    config.log_dir = tempdir
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ["sine.wav"]
    audio_io.save(os.path.join(tempdir, "sine.wav"), x, sample_rate=config.sample_rate)
    config.cond_channels = 20
    config.n_mels = 20
    config.dataset_compute_mel = True

    trainer = trainer_class(config=config, device=DEVICE)
    trainer.fit(num_iters=1)
    y = trainer.generate(trainer.create_dataset_training())
    assert y.device.type == 'cpu'
    assert y.shape == (1, 1, config.segment_len)
    assert torch.isfinite(y).all()
