from glob import glob
from logging import warning
import os
import torchaudio
from typing import List, Tuple, Union
import torch
from torch.utils.data import Dataset

import numpy as np

from glotnet.config import Config

from glotnet.sigproc.melspec import SpectralNormalization, InverseSpectralNormalization
from glotnet.sigproc.melspec import LogMelSpectrogram

class AudioDataset(Dataset):
    """ Dataset for audio files """

    def __init__(self,
                 config: Config,
                 audio_dir: str,
                 audio_ext: str = '.wav',
                 file_list: List[str] = None,
                 transforms: Union[torch.nn.Module, str] = None,
                 dtype: torch.dtype = torch.float32):
        """
        Args:
            config: Config object
            audio_dir: directory containing audio files
            audio_ext: file extension of audio files
            file_list: list of audio files
            transforms: transforms to apply to audio, output as auxiliary feature for conditioning
            dtype: data type of output
        """

        self.config = config
        self.audio_dir = audio_dir
        self.audio_ext = audio_ext
        self.dtype = dtype
        self.transforms = transforms

        self.use_scaler = False

        if config.dataset_scaler_file is not None:
            data = np.load(config.dataset_scaler_file)
            self.scaler_m = torch.tensor(data['mean'], dtype=torch.float32).reshape(1, -1, 1)
            self.scaler_s = torch.tensor(data['scale'], dtype=torch.float32).reshape(1, -1, 1)
            self.use_scaler = True

        if file_list is None:
            self.audio_files = glob(os.path.join(
                self.audio_dir, f"*{self.audio_ext}"))
        else:
            self.audio_files = file_list

        # elements are (filename, start, stop)
        self.segment_index: List[Tuple(str, int, int)] = []

        for f in self.audio_files:
            self._check_audio_file(f)

    @staticmethod
    def read_filelist(audio_dir:str, filelist:Union[str, list]):
        """ Read a list of files from a file or a list of files """

        file_list = []
        # if filelist is a list, then it is already a list of files
        if type(filelist) == list:
            for f in filelist:
                file_list.append(os.path.join(audio_dir, f.strip()))
            return file_list
        # if filelist is a string, then it is a path to a file containing a list of files
        with open(filelist, 'r') as f:
            for line in f.readlines():
                file_list.append(os.path.join(audio_dir, line.strip()))
        return file_list


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
        
        if self.transforms is not None:
            c = self.transforms(x)
            if self.use_scaler:
                c = (c - self.scaler_m) / self.scaler_s
            c = c[0] # drop batch dimension, DataLoader will put it back
            return (x, c)
        else:
            return (x,)
