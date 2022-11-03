import torch

from .lfilter import LFilter
from typing import Tuple

class BiquadBandPass(LFilter):

    def __init__(self, freq: float = 0.5, gain: float = 0.0, Q: float = 0.7071):
        """
        Args:
            freq: normalized frequency from 0 to 1 (Nyquist)
            gain: gain in dB
            Q: quality factor determining filter resonance bandwidth
        """
        # TODO: pass STFT arguments to LFilter
        super().__init__(n_fft=2048, hop_length=512, win_length=1024)

        self.gain_dB = torch.nn.Parameter(torch.tensor(gain, dtype=torch.float32).reshape(1, 1))
        self.freq = torch.nn.Parameter(torch.tensor(freq, dtype=torch.float32).reshape(1, 1))
        self.Q = torch.nn.Parameter(torch.tensor(Q, dtype=torch.float32).reshape(1, 1))

    def _params_to_direct_form(self,
                               freq: torch.Tensor,
                               gain: torch.Tensor,
                               Q: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        
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
                freq: torch.Tensor = None,
                gain: torch.Tensor = None,
                Q: torch.Tensor = None,
                ) -> torch.Tensor:

        if freq is None:
            freq = self.freq
        if gain is None:
            gain = self.gain_dB
        if Q is None:
            Q = self.Q

        b, a = self._params_to_direct_form(freq=freq, gain=gain, Q=Q)
        num_frames = x.size(-1) // self.hop_length
        if b.ndim < 3:
            b = b.reshape(1, -1, 1).expand(-1, -1, num_frames)
        if a.ndim < 3:
            a = a.reshape(1, -1, 1).expand(-1, -1, num_frames)
        

        return super().forward(x, b=b, a=a)




