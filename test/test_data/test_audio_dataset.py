from glotnet.data.audio_dataset import AudioDataset
from glotnet.data.config import DataConfig

import soundfile as sf
import os
import tempfile
import numpy as np

def test_directory_read():

    config = DataConfig(padding=1000)

    num_files = 3
    padded_seqment_len = config.segment_len + config.padding
    ext = '.wav'

    with tempfile.TemporaryDirectory() as dir:
        for i in range(num_files):
            num_samples = np.random.randint(
                low=int(0.1 * config.sample_rate),
                high=int(8 * config.sample_rate))
            x = np.random.randn(num_samples, config.channels)

            sf.write(os.path.join(dir, f"{i}{ext}"),
                     x, samplerate=config.sample_rate)

        dataset = AudioDataset(config, dir, ext)

        for i, d in enumerate(dataset):
            assert d.size(-1) == padded_seqment_len, \
                f"all items in dataset should be of length {padded_seqment_len}, got {d.size(-1)} in element {i}"


if __name__ == "__main__":
    test_directory_read()