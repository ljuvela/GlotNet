""" Audio file I/O based on soundfile

Replaces torchaudio.info / load / save, which were removed or moved to
torchcodec in recent torchaudio releases. Tensors use the torchaudio
convention of shape (channels, time).
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import soundfile as sf
import torch


@dataclass
class AudioInfo:
    sample_rate: int
    num_frames: int
    num_channels: int


def info(path: str) -> AudioInfo:
    """ Read audio file metadata """
    i = sf.info(path)
    return AudioInfo(sample_rate=i.samplerate,
                     num_frames=i.frames,
                     num_channels=i.channels)


def load(path: str, frame_offset: int = 0,
         num_frames: int = -1) -> Tuple[torch.Tensor, int]:
    """ Load audio as float32 tensor of shape (channels, time) """
    x, sample_rate = sf.read(path, start=frame_offset, frames=num_frames,
                             dtype='float32', always_2d=True)
    return torch.from_numpy(np.ascontiguousarray(x.T)), sample_rate


def save(path: str, x: torch.Tensor, sample_rate: int,
         subtype: Optional[str] = None):
    """ Save tensor of shape (channels, time) or (time,) to audio file

    If subtype is None, floating point tensors are written as 32-bit float
    (matching torchaudio.save defaults), otherwise the soundfile default
    for the file format is used.
    """
    x = x.detach().cpu()
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if subtype is None and x.is_floating_point() \
            and path.lower().endswith('.wav'):
        subtype = 'FLOAT'
    sf.write(path, x.numpy().T, sample_rate, subtype=subtype)
