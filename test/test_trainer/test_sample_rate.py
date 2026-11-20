""" Sample rate comes from the config (e.g. 24 kHz for LibriTTS-R) """
import os
import pytest
import torch
from glotnet.data import audio_io

from glotnet.config import Config
from glotnet.sigproc.melspec import LogMelSpectrogram
from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.trainer.trainer_glotnet import TrainerGlotNet


def make_config(tempdir, sample_rate, **kwargs):
    config = Config(
        sample_rate=sample_rate,
        batch_size=2,
        learning_rate=1e-4,
        dilations=[1, 2],
        residual_channels=4,
        skip_channels=4,
        segment_len=1000,
        cond_channels=20,
        n_mels=20,
        dataset_compute_mel=True,
        log_dir=tempdir,
        dataset_audio_dir_training=tempdir,
        dataset_filelist_training=["sine.wav"],
        **kwargs)
    seg_len = config.batch_size * config.segment_len
    t = torch.arange(seg_len) / sample_rate
    x = 0.5 * torch.sin(2 * torch.pi * 200 * t).unsqueeze(0)
    audio_io.save(os.path.join(tempdir, "sine.wav"), x, sample_rate=sample_rate)
    return config


def test_mel_fmax_default_is_nyquist():
    assert Config().mel_fmax is None
    # at 16 kHz the default gives the same filterbank as the old mel_fmax = 8000
    fb_default = LogMelSpectrogram(sample_rate=16000, f_max=None).mel_spectrogram.mel_scale.fb
    fb_8k = LogMelSpectrogram(sample_rate=16000, f_max=8000).mel_spectrogram.mel_scale.fb
    assert torch.equal(fb_default, fb_8k)


@pytest.mark.parametrize("trainer_class, kwargs", [
    (TrainerWaveNet, {}),
    (TrainerGlotNet, {'input_channels': 3}),
])
def test_trainer_24khz_fit_and_generate(tempdir, trainer_class, kwargs):
    config = make_config(tempdir, 24000, **kwargs)
    trainer = trainer_class(config=config)
    trainer.fit(num_iters=2)
    assert torch.isfinite(trainer.batch_loss)

    dataset = trainer.create_dataset_training()
    mel = dataset.transforms.mel_spectrogram
    assert mel.sample_rate == 24000
    assert mel.f_max is None  # torchaudio then uses sample_rate / 2
    # the top mel band reaches above the old fixed 8 kHz limit
    fb = mel.mel_scale.fb  # (n_freqs, n_mels)
    freqs = torch.linspace(0, 12000, fb.size(0))
    assert freqs[fb[:, -1] > 0].max() > 8000

    x = trainer.generate(dataset=dataset)
    assert x.shape == (1, 1, config.segment_len)
    assert torch.isfinite(x).all()


def test_dataset_rejects_other_sample_rate(tempdir):
    config = make_config(tempdir, 24000)
    config.sample_rate = 16000
    with pytest.raises(ValueError, match="sample rate"):
        TrainerWaveNet(config=config)


def test_checkpoint_sample_rate_mismatch(tempdir):
    config = make_config(tempdir, 24000)
    trainer = TrainerWaveNet(config=config)
    model_pt = os.path.join(tempdir, "model.pt")
    trainer.save(model_path=model_pt)
    assert TrainerWaveNet.config_from_checkpoint(model_pt).sample_rate == 24000

    other_dir = os.path.join(tempdir, "16k")
    os.makedirs(other_dir)
    config_16k = make_config(other_dir, 16000)
    with pytest.raises(ValueError, match="sample_rate"):
        TrainerWaveNet(config=config_16k).load(model_pt)
