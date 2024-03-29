import os
import tempfile
import numpy as np
import torch
import torchaudio
import pytest

from torch.utils.data.dataloader import DataLoader

from glotnet.data.audio_dataset import AudioDataset
from glotnet.config import Config
from glotnet.sigproc.melspec import LogMelSpectrogram

# TODO pytest fixture tempdir
# @pytest.fixture(scope="session")
# def generate_random_wav_files(tmp_path_factory, num_files, sample_rate, channels, ext='.wav'):


def generate_random_wav_files(dir, num_files, sample_rate, channels, ext='.wav'):
    """ Generate random wave files for testing """
    for i in range(num_files):
        num_samples = np.random.randint(
            low=int(0.1 * sample_rate),
            high=int(8 * sample_rate))
        x = torch.randn(channels, num_samples)
        torchaudio.save(os.path.join(dir, f"{i}{ext}"),
                    x, sample_rate=sample_rate)

def test_padding():
    config = Config(padding=1000)
    num_files = 3
    padded_seqment_len = config.segment_len + config.padding
    ext = '.wav'
    with tempfile.TemporaryDirectory() as dir:
        generate_random_wav_files(dir, num_files, config.sample_rate, config.channels)
        dataset = AudioDataset(config, dir, ext)
        # Test that each dataset sample is padded to expected length
        for i, (d,) in enumerate(dataset):
            assert d.size(-1) == padded_seqment_len, \
                f"all items in dataset should be of length {padded_seqment_len}, got {d.size(-1)} in element {i}"

def test_batching():
    config = Config(padding=1000)
    num_files = 3
    ext = '.wav'

    batch_size = 4
    channels = 1
    config.batch_size = batch_size
    config.channels = channels
    padded_seqment_len = config.segment_len + config.padding
    
    with tempfile.TemporaryDirectory() as dir:
        generate_random_wav_files(dir, num_files, config.sample_rate, config.channels)
        dataset = AudioDataset(config, dir, ext)
        # Test that batch elements are correct size
        dataloader = DataLoader(dataset=dataset,
                                batch_size=config.batch_size,
                                shuffle=True, drop_last=True)

        for minibatch in dataloader:
            x, = minibatch
            assert x.shape == (batch_size, channels, padded_seqment_len), \
                f"Expected minibatch size {(batch_size, channels, padded_seqment_len)}, got {x.shape}"
            


def test_mel_extraction():
    config = Config(padding=1000)
    num_files = 3
    ext = '.wav'

    batch_size = 4
    channels = 1
    config.batch_size = batch_size
    config.channels = channels
    padded_seqment_len = config.segment_len + config.padding
    
    melspec =  LogMelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        f_min=config.mel_fmin,
        f_max=config.mel_fmax,
        n_mels=config.n_mels,
    )

    with tempfile.TemporaryDirectory() as dir:
        generate_random_wav_files(dir, num_files, config.sample_rate, config.channels)
        dataset = AudioDataset(config, dir, ext, transforms=melspec)
        # Test that batch elements are correct size
        dataloader = DataLoader(dataset=dataset,
                                batch_size=config.batch_size,
                                shuffle=True, drop_last=True)

        for minibatch in dataloader:
            x, c = minibatch
            assert x.shape == (batch_size, channels, padded_seqment_len), \
                f"Expected audio size {(batch_size, channels, padded_seqment_len)}, got {x.shape}"
            print(f"cond shape {c.shape} ")

            shape_ref = (batch_size, config.n_mels, padded_seqment_len // config.hop_length + 1)
            assert c.shape == shape_ref, \
                f"Expected mel shape {shape_ref}, got {c.shape}"


if __name__ == "__main__":
    test_padding()
    test_batching()
    test_mel_extraction()