from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TrainerConfig:
    """ GlotNet Trainer configuration """

    # Model
    skip_channels: int = 32
    residual_channels: int = 32
    filter_width: int = 3
    dilations: Tuple[int] = (1, 2, 4, 8, 16)
    input_channels: int = 1
    activation: str = "gated"
    use_residual: bool = True,
    cond_channels: int = None

    # Distribution
    distribution: str = 'gaussian'

    # Optimizer
    optimizer: str = 'adam'
    learning_rate: float = 1e-3
