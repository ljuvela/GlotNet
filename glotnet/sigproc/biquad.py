import torch

from .lfilter import LFilter
from typing import Tuple


class BiquadBandPassFunctional(torch.nn.Module):

    def __init__(self):

        super().__init__()

        # TODO: pass STFT arguments to LFilter
        self.n_fft = 2048
        self.hop_length = 512
        self.win_length = 1024
        self.lfilter = LFilter(n_fft=self.n_fft,
                               hop_length=self.hop_length,
                               win_length=self.win_length)

    def _params_to_direct_form(self,
                               freq: torch.Tensor,
                               gain: torch.Tensor,
                               Q: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            freq, center frequency, shape is (..., 1)
            gain, gain in decibels, shape is (..., 1)
            Q, resonance sharpness, shape is (..., 1)
        """
        if torch.any(freq > 1.0):
            raise ValueError(f"Normalized frequency must be below 1.0, max was {freq.max()}")
        if torch.any(freq < 0.0):
            raise ValueError(f"Normalized frequency must be above 0.0, min was {freq.min()}")
        omega = torch.pi * freq
        A = torch.pow(10.0, 0.025 * gain)
        alpha = 0.5 * torch.sin(omega) / Q

        b0 = 1.0 + alpha * A
        b1 = -2.0 * torch.cos(omega)
        b2 = 1.0 - alpha * A
        a0 = 1 + alpha / A
        a1 = -2.0 * torch.cos(omega)
        a2 = 1 - alpha / A

        a = torch.cat([a0, a1, a2], dim=-1)
        b = torch.cat([b0, b1, b2], dim=-1)
        b = b / a0
        a = a / a0
        return b, a

    def forward(self,
                x: torch.Tensor,
                freq: torch.Tensor,
                gain: torch.Tensor,
                Q: torch.Tensor,
                ) -> torch.Tensor:
        """ 
        Args:
            x: input signal
                shape = (batch, channels, time)
            freq: center frequencies 
                shape = (batch, channels, n_filters, n_frames)
                n_frames should be time // hop_size
            gain: gains in decibels,
                shape = (batch, channels, n_filters, n_frames)
            Q: filter resonance (quality factor)
                shape = (batch, channels, n_filters, n_frames)
        """



        b, a = self._params_to_direct_form(freq=freq, gain=gain, Q=Q)
        num_frames = x.size(-1) // self.hop_length
        if b.ndim < 3:
            b = b.reshape(1, -1, 1).expand(-1, -1, num_frames)
        if a.ndim < 3:
            a = a.reshape(1, -1, 1).expand(-1, -1, num_frames)
        
        # reshape parallel filters to batch

        # expand input for parallel filters

        y = self.lfilter.forward(x, b=b, a=a)

        return y


class BiquadBandPassModule(BiquadBandPassFunctional):

    def __init__(self, freq: float = 0.5, gain: float = 0.0, Q: float = 0.7071):
        """
        Args:
            freq: normalized frequency from 0 to 1 (Nyquist)
            gain: gain in dB
            Q: quality factor determining filter resonance bandwidth
        """
        super().__init__()

        # TODO: separate wrapper module for self-contained parameters (inherit biquad functional?)
        self.gain_dB = torch.nn.Parameter(torch.tensor(gain, dtype=torch.float32).reshape(1, 1))
        self.freq = torch.nn.Parameter(torch.tensor(freq, dtype=torch.float32).reshape(1, 1))
        self.Q = torch.nn.Parameter(torch.tensor(Q, dtype=torch.float32).reshape(1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        freq = self.freq
        gain = self.gain_dB
        Q = self.Q

        # TODO reshape inputs

        y = super.forward(x, freq, gain, Q)

        # TODO reshape outputs

        return y