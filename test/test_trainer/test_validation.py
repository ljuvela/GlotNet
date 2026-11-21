""" Validation iterates the validation data (not the training data), every segment once """
import os
import pytest
import torch
from glotnet.data import audio_io

from glotnet.config import Config
from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.trainer.trainer_glotnet import TrainerGlotNet


@pytest.mark.parametrize("trainer_class, kwargs", [
    (TrainerWaveNet, {}),
    (TrainerGlotNet, {'input_channels': 3}),
])
def test_validation_uses_validation_data(tempdir, trainer_class, kwargs):
    config = Config(batch_size=4, segment_len=1000, dilations=[1, 2],
                    residual_channels=4, skip_channels=4, log_dir=tempdir,
                    cond_channels=20, n_mels=20, dataset_compute_mel=True, **kwargs)
    t = torch.arange(20000) / config.sample_rate
    train = 0.5 * torch.sin(2 * torch.pi * 200 * t).unsqueeze(0)
    # different length and content: 7 segments, not a multiple of batch_size
    val = 0.1 * torch.randn(1, 7 * config.segment_len)
    audio_io.save(os.path.join(tempdir, "train.wav"), train, sample_rate=config.sample_rate)
    audio_io.save(os.path.join(tempdir, "val.wav"), val, sample_rate=config.sample_rate)
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ["train.wav"]
    config.dataset_audio_dir_validation = tempdir
    config.dataset_filelist_validation = ["val.wav"]

    trainer = trainer_class(config=config)
    assert len(trainer.dataset_validation) == 7
    assert len(trainer.dataset_training) == 20
    loader = trainer.data_loader_validation
    assert loader.dataset is trainer.dataset_validation

    # every validation segment exactly once, in order, the last batch included
    batches = [x for x, c in loader]
    assert [b.size(0) for b in batches] == [4, 3]
    x_all = torch.cat(batches)
    for i in range(len(trainer.dataset_validation)):
        assert torch.equal(x_all[i], trainer.dataset_validation[i][0])

    loss = trainer.validate()
    assert torch.isfinite(torch.tensor(loss))
