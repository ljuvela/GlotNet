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
    loss_weight_scale_hinge_reg: float = 0.1

    # Optimizer
    optimizer: str = 'adam'
    learning_rate: float = 1e-3
    batch_size: int = 4
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
    hop_size: int = 256
    frame_size: int = 1024
    nfft: int = 1024
    num_mels: int = 80

    # summary writer 
    log_dir: str = 'runs/logs'

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