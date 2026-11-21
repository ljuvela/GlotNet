""" Trainer.generate over several dataset items as one batch """
import os
import pytest
import torch
from glotnet.data import audio_io

from glotnet.config import Config
from glotnet.train import generation_indices
from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.trainer.trainer_glotnet import TrainerGlotNet


def make_trainer(tempdir, trainer_class, **kwargs):
    config = Config(batch_size=2, segment_len=600, dilations=[1, 2],
                    residual_channels=4, skip_channels=4, log_dir=tempdir,
                    pre_emphasis=0.85, cond_channels=20, n_mels=20,
                    dataset_compute_mel=True, **kwargs)
    torch.manual_seed(0)
    t = torch.arange(6 * config.segment_len) / config.sample_rate
    f0 = 100 + 50 * t  # differs from segment to segment
    x = 0.5 * torch.sin(2 * torch.pi * f0 * t).unsqueeze(0) + 0.01 * torch.randn(1, t.numel())
    audio_io.save(os.path.join(tempdir, "x.wav"), x, sample_rate=config.sample_rate)
    config.dataset_audio_dir_training = tempdir
    config.dataset_filelist_training = ["x.wav"]
    trainer = trainer_class(config=config)
    trainer.fit(num_iters=2)
    return trainer


@pytest.mark.parametrize("trainer_class, kwargs", [
    (TrainerWaveNet, {}),
    (TrainerGlotNet, {'input_channels': 3}),
])
def test_generate_batch_equals_items(tempdir, trainer_class, kwargs):
    trainer = make_trainer(tempdir, trainer_class, **kwargs)
    data = trainer.create_dataset_training()
    indices = [0, 3, 5]
    # temperature 0: greedy, so batch items do not depend on the RNG stream
    greedy = dict(temperature_voiced=0.0, temperature_unvoiced=0.0)
    batch = trainer.generate(data, indices=indices, **greedy)
    assert batch.shape == (3, 1, trainer.config.segment_len)
    assert torch.isfinite(batch).all()
    for k, i in enumerate(indices):
        single = trainer.generate(data, indices=[i], **greedy)
        assert torch.allclose(batch[k], single[0], atol=1e-6), f"item {i}"
    assert not torch.allclose(batch[0], batch[1]), "items must differ"

    # default: one item (old behaviour), temperature 1
    x = trainer.generate(data)
    assert x.shape == (1, 1, trainer.config.segment_len)


def test_generation_indices():
    assert generation_indices(100, 4) == [0, 33, 66, 99]
    assert generation_indices(3, 8) == [0, 1, 2]
    assert generation_indices(10, 1) == [0]
