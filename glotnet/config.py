from dataclasses import asdict, dataclass
import json
from typing import List, Tuple

import torch

@dataclass
class Config:
    """ GlotNet configuration """

    # Model
    input_channels: int = 1
    skip_channels: int = 32
    residual_channels: int = 32
    filter_width: int = 3
    dilations: Tuple[int] = (1, 2, 4, 8, 16)
    activation: str = "gated"
    use_residual: bool = True
    cond_channels: int = None

    # Distribution
    distribution: str = 'gaussian'
    loss_weight_nll: float = 1.0
    loss_weight_entropy_hinge: float = 0.1
    entropy_floor: float = -9.0

    # Optimizer
    optimizer: str = 'adam'
    learning_rate: float = 1e-3
    batch_size: int = 4
    dataloader_workers: int = 0
    shuffle: bool = True

    # Training
    max_iters: int = 10 ** 6
    logging_interval: int = 10 ** 2
    validation_interval: int = 10 ** 3

    # Audio properties
    sample_rate: int = 16000
    channels: int = 1
    segment_len: int = 8000
    padding: int = 0
    # acoustic features
    hop_length: int = 256
    win_length: int = 1024
    n_fft: int = 1024
    n_mels: int = 80
    mel_fmin: float = 0.0
    mel_fmax: float = 8000

    # from https://github.com/pytorch/audio/blob/6b2b6c79ca029b4aa9bdb72d12ad061b144c2410/examples/pipeline_tacotron2/train.py#L180
    # audio = parser.add_argument_group("audio parameters")
    # audio.add_argument("--sample-rate", default=22050, type=int, help="Sampling rate")
    # audio.add_argument("--n-fft", default=1024, type=int, help="Filter length for STFT")
    # audio.add_argument("--hop-length", default=256, type=int, help="Hop (stride) length")
    # audio.add_argument("--win-length", default=1024, type=int, help="Window length")
    # audio.add_argument("--n-mels", default=80, type=int, help="")
    # audio.add_argument("--mel-fmin", default=0.0, type=float, help="Minimum mel frequency")
    # audio.add_argument("--mel-fmax", default=8000.0, type=float, help="Maximum mel frequency")


    # summary writer 
    log_dir: str = None

    # saving
    saves_dir: str = 'runs/saves'
    
    @property
    def segment_dur(self) -> float:
        return float(self.segment_len) / float(self.sample_rate)

    @segment_dur.setter
    def segment_dur(self, dur: float) -> None:
        self.segment_len = int(dur * self.sample_rate)

    def from_json(filepath: str):
        """ Load Config from Json"""
        with open(filepath, 'r') as f:
            cfg = json.load(f)
        return Config(**cfg)

    def to_json(self, filepath: str):
        "Write Config to JSON"
        with open(filepath, 'w') as f:
            json.dump(asdict(self), fp=f, indent=4)