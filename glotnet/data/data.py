import torch

from torch.utils.data import Dataset
from torchaudio.transforms import Resample

class NeuralVocoderDataset(Dataset):

    def __init__(self, dataset, sequence_len, warmup=0, sample_rate=16000):
        """ Dataset wrapper for training neural vocoders

        Args:
            dataset: torchaudio dataset instance
            sequence_len: length of output sequences
        """
        self.dataset = dataset
        self.sequence_len = sequence_len
        self.warmup = warmup 


    def _get_resampler(self):
        # TODO: get one element from dataset and make a resampler
        self.resampler = Resample()

    def __getitem__(self, i):

        item = self.dataset.__getitem__(i)
        x = item[0] # take raw waveform only

        x = torch.permute(x, dims=(1, 0)) # (C, T) -> (T, C)
        max_ind = x.shape[0] - self.sequence_len
        if max_ind < 0:
            # TODO: zero pad
            raise ValueError(f"Signal length {x.shape[0]} is shorter than sequence length {self.sequence_len}")
        start = torch.randint(low=self.warmup, high=max_ind, size=(1,))
        stop = start + self.sequence_len

        segment = x[start:stop, :]
        return segment

    def __len__(self):
        return len(self.dataset)
