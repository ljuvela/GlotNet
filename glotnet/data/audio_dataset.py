from glob import glob
from logging import warn
import os
import soundfile as sf
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

from .config import DataConfig

class AudioDataset(Dataset):

    def __init__(self,
                 config: DataConfig,
                 audio_dir: str,
                 audio_ext: str = '.wav',
                 file_list: List[str] = None,
                 dtype: torch.dtype = torch.float32):
        self.config = config
        self.audio_dir = audio_dir
        self.audio_ext = audio_ext
        self.dtype = dtype

        if file_list is None:
            self.audio_files = glob(os.path.join(
                self.audio_dir, f"*{self.audio_ext}"))
        else:
            self.audio_files = file_list

        # elements are (filename, start, stop)
        self.segment_index: List[Tuple(str, int, int)] = []

        for f in self.audio_files:
            self._check_audio_file(f)

    def _check_audio_file(self, f):

        info = sf.info(f)
        if info.channels != self.config.channels:
            raise ValueError(
                f"Expected {self.config.channels} but got {info.channels} in {f}")
        if info.samplerate != self.config.sample_rate:
            raise ValueError(
                f"Expected sample rate {self.config.sample_rate} but got {info.samplerate} in {f}")

        num_samples = info.frames

        if num_samples < self.config.segment_len:
            warn(
                f"File {f} is shorter than specified segment length {self.config.segment_len}, was {num_samples}")

        # read sements from the end (to minimize zero padding)
        stop = num_samples
        start = stop - self.config.segment_len - self.config.padding
        while stop >= 0:
            self.segment_index.append(
                (os.path.realpath(f), start, stop))
            stop = stop - self.config.segment_len
            start = stop - self.config.segment_len - self.config.padding
            start = max(start, 0)

    def __len__(self):
        return len(self.segment_index)

    def __getitem__(self, i):
        f, start, stop = self.segment_index[i]
        x, fs = sf.read(f, start=start, stop=stop)
        
        x = torch.tensor(x, dtype=self.dtype)
        if x.ndim == 1:
            x = x.unsqueeze(0) # shape is (channels=1, time)
        x = x.unsqueeze(0) # shape is (batch=1, channels, time)
        pad_left = self.config.segment_len + self.config.padding - x.size(-1)
        # zero pad to segment_len + padding
        x = torch.nn.functional.pad(x, (pad_left, 0))
        return x
