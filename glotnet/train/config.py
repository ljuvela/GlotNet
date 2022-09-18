from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class TrainerConfig:
    """ GlotNet Trainer configuration """

    # Model
    input_channels: int = 1
    skip_channels: int = 32
    residual_channels: int = 32
    filter_width: int = 3
    dilations: Tuple[int] = (1, 2, 4, 8, 16)
    activation: str = "gated"
    use_residual: bool = True,
    cond_channels: int = None

    # Distribution
    distribution: str = 'gaussian'

    # Optimizer
    optimizer: str = 'adam'
    learning_rate: float = 1e-3
    batch_size: int = 4
    shuffle: bool = True

    loss_weight_nll: float = 1.0
    loss_weight_scale_hinge_reg: float = 0.1
