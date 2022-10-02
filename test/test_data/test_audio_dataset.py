import os
import tempfile
import numpy as np
import torch
import torchaudio

from torch.utils.data.dataloader import DataLoader

from glotnet.data.audio_dataset import AudioDataset
from glotnet.config import Config

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
            


if __name__ == "__main__":
    test_padding()
    test_batching()