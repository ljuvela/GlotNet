from glob import glob
from logging import warning
import os
import torchaudio
from typing import List, Tuple
import torch
from torch.utils.data import Dataset

from glotnet.config import Config

from glotnet.sigproc.melspec import SpectralNormalization, InverseSpectralNormalization
from glotnet.sigproc.melspec import LogMelSpectrogram

class AudioDataset(Dataset):

    def __init__(self,
                 config: Config,
                 audio_dir: str,
                 audio_ext: str = '.wav',
                 file_list: List[str] = None,
                 output_mel: bool = False, # TODO: take transform as argument
                 dtype: torch.dtype = torch.float32):
        self.config = config
        self.audio_dir = audio_dir
        self.audio_ext = audio_ext
        self.output_mel = output_mel
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

        # TODO: use sigproc.melspec.LogMelSpectrogram
        #https://github.com/pytorch/audio/blob/6b2b6c79ca029b4aa9bdb72d12ad061b144c2410/examples/pipeline_tacotron2/train.py#L284
        self.transforms = torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=config.sample_rate,
                n_fft=config.n_fft,
                win_length=config.win_length,
                hop_length=config.hop_length,
                f_min=config.mel_fmin,
                f_max=config.mel_fmax,
                n_mels=config.n_mels,
                mel_scale="slaney",
                normalized=False,
                power=1,
                norm="slaney",
            ),
            SpectralNormalization(),
        )

    def _check_audio_file(self, f):

        info = torchaudio.info(f)
        if info.num_channels != self.config.channels:
            raise ValueError(
                f"Expected {self.config.channels} but got {info.num_channels} in {f}")
        if info.sample_rate != self.config.sample_rate:
            raise ValueError(
                f"Expected sample rate {self.config.sample_rate} but got {info.sample_rate} in {f}")

        num_samples = info.num_frames

        if num_samples < self.config.segment_len:
            warning(
                f"File {f} is shorter than specified segment length {self.config.segment_len}, was {num_samples}")

        # read sements from the end (to minimize zero padding)
        stop = num_samples
        start = stop - self.config.segment_len - self.config.padding
        start = max(start, 0)
        while stop > 0:
            self.segment_index.append(
                (os.path.realpath(f), start, stop))
            stop = stop - self.config.segment_len
            start = stop - self.config.segment_len - self.config.padding
            start = max(start, 0)

    def __len__(self):
        return len(self.segment_index)

    def __getitem__(self, i):
        f, start, stop = self.segment_index[i]
        x, fs = torchaudio.load(f, frame_offset=start, num_frames=stop-start)
        if x.ndim == 1:
            x = x.unsqueeze(0) # shape is (channels=1, time)
        pad_left = self.config.segment_len + self.config.padding - x.size(-1)
        # zero pad to segment_len + padding
        x = torch.nn.functional.pad(x, (pad_left, 0))
        
        if self.output_mel:
            c = self.transforms(x)
            c = c[0] # drop batch dimension, DataLoader will put it back
            return (x, c)
        else:
            return (x,)
