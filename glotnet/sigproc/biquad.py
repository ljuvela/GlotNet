import torch

from .lfilter import LFilter
from typing import Tuple, Union


class BiquadBaseFunctional(torch.nn.Module):

    def __init__(self):
        """ Initialize Biquad"""

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
            freq, center frequency,
                shape is (batch, 1, n_frames)
            gain, gain in decibels
                shape is (batch, 1, n_frames)
            Q, resonance sharpness
                shape is (batch, 1, n_frames)

        Returns:
            a, filter denominator coefficients
                shape is (batch, n_taps, n_frames)

        """
        raise NotImplementedError("Subclasses must implement this method")

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
                n_frames is expected to be (time // hop_size)
            gain: gains in decibels,
                shape = (batch, channels, n_filters, n_frames)
            Q: filter resonance (quality factor)
                shape = (batch, channels, n_filters, n_frames)
        """

        # save parameter shapes for later
        batch, channels, n_filters, n_frames = freq.shape

        # reshape to (batch * channels * n_filters, 1, n_frames)
        freq = freq.reshape(-1, 1, freq.size(-1))
        gain = gain.reshape(-1, 1, gain.size(-1))
        Q = Q.reshape(-1, Q.size(-1))

        b, a = self._params_to_direct_form(freq=freq, gain=gain, Q=Q)

        x = x.reshape(batch * channels * n_filters, 1, -1)

        y = self.lfilter.forward(x, b=b, a=a)

        # reshape to (batch, channels, n_filters, time)
        y = y.reshape(batch, channels, n_filters, -1)

        return y
    

class BiquadPeakFunctional(BiquadBaseFunctional):

    def __init__(self):
        """ Initialize Biquad"""
        super().__init__()

    def _params_to_direct_form(self,
                               freq: torch.Tensor,
                               gain: torch.Tensor,
                               Q: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            freq, center frequency,
                shape is (batch, 1, n_frames)
            gain, gain in decibels
                shape is (batch, 1, n_frames)
            Q, resonance sharpness
                shape is (batch, 1, n_frames)

        Returns:
            a, filter denominator coefficients
                shape is (batch, n_taps, n_frames)

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

        a = torch.cat([a0, a1, a2], dim=1)
        b = torch.cat([b0, b1, b2], dim=1)

        b = b / a0
        a = a / a0
        return b, a


class BiquadModule(torch.nn.Module):

    def __init__(self,
                 freq: Union[torch.Tensor, float] = torch.tensor([0.5]),
                 gain: Union[torch.Tensor, float] = torch.tensor([0.0]),
                 Q: Union[torch.Tensor, float] = torch.tensor([0.7071]),
                 fs: float = None,
                 func: BiquadBaseFunctional = BiquadPeakFunctional()
                ):
        """
        Args:
            func: BiquadBaseFunctional subclass
            freq: center frequency 
            gain: gain in dB
            Q: quality factor determining filter resonance bandwidth
            fs: sample rate, if not provided freq is assumed as normalized from 0 to 1 (Nyquist)

        """
        super().__init__()

        self.func = func

        # if no sample rate provided, assume normalized frequency
        if fs is None:
            fs = 2.0

        self.fs = fs

        if type(freq) != torch.Tensor:
            freq = torch.tensor([freq], dtype=torch.float32)
        if type(gain) != torch.Tensor:
            gain = torch.tensor([gain], dtype=torch.float32)
        if type(Q) != torch.Tensor:
            Q = torch.tensor([Q], dtype=torch.float32)

        # convert to normalized frequency
        freq = 2.0 * freq / fs

        if freq.max() > 1.0:
            raise ValueError(
                "Maximum normalized frequency is larger than 1.0. "
                "Please provide a sample rate or input normalized frequencies")
        if freq.min() < 0.0:
            raise ValueError(
                "Maximum normalized frequency is smaller than 0.0.")

        # reshape to (batch, channels, n_filters)
        self.gain_dB = torch.nn.Parameter(gain.reshape(1, 1, -1))
        self.freq = torch.nn.Parameter(freq.reshape(1, 1, -1))
        self.Q = torch.nn.Parameter(Q.reshape(1, 1, -1))

    def get_impulse_response(self, n_timesteps: int = 2048) -> torch.Tensor:
        """ Get impulse response of filter

        Args:
            n_timesteps: number of timesteps to evaluate

        Returns:
            h, shape is (batch, channels, n_timesteps)
        """
        x = torch.zeros(1, 1, n_timesteps)
        x[:, :, 0] = 1.0
        h = self.forward(x)
        return h
    
    def get_frequency_response(self, n_timesteps: int = 2048, n_fft: int = 2048) -> torch.Tensor:
        """ Get frequency response of filter

        Args:
            n_timesteps: number of timesteps to evaluate

        Returns:
            H, shape is (batch, channels, n_timesteps)
        """
        h = self.get_impulse_response(n_timesteps=n_timesteps)
        H = torch.fft.rfft(h, n=n_fft, dim=-1)
        H = torch.abs(H)
        return H


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, shape is (batch, channels, timesteps)

        Returns:
            y, shape is (batch, channels, timesteps)
        """
        freq = self.freq
        gain = self.gain_dB
        Q = self.Q

        # Expand time dimension
        # TODO: separate backend for time-constant params
        timesteps = x.size(2)
        num_frames = timesteps // self.func.hop_length
        freq = freq.unsqueeze(-1).expand(-1, -1, -1, num_frames)
        gain = gain.unsqueeze(-1).expand(-1, -1, -1, num_frames)
        Q = Q.unsqueeze(-1).expand(-1, -1, -1, num_frames)

        # reshape parameters
        # freq.expand()

        y = self.func.forward(x, freq, gain, Q)

        # TODO reshape outputs

        return y



class BiquadResonatorFunctional(BiquadBaseFunctional):

    def __init__(self):
        """ Initialize Biquad"""
        super().__init__()

    def _params_to_direct_form(self,
                               freq: torch.Tensor,
                               gain: torch.Tensor,
                               Q: torch.Tensor
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args: 
            freq, center frequency,
                shape is (batch, 1, n_frames)
            gain, gain in decibels
                shape is (batch, 1, n_frames)
            Q, resonance sharpness
                shape is (batch, 1, n_frames)

        Returns:
            a, filter denominator coefficients
                shape is (batch, n_taps, n_frames)

        """
        if torch.any(freq > 1.0):
            raise ValueError(f"Normalized frequency must be below 1.0, max was {freq.max()}")
        if torch.any(freq < 0.0):
            raise ValueError(f"Normalized frequency must be above 0.0, min was {freq.min()}")
        omega = torch.pi * freq
        A = torch.pow(10.0, 0.025 * gain)
        alpha = 0.5 * torch.sin(omega) / Q

        b0 = torch.ones_like(freq)
        b1 = torch.zeros_like(freq)
        b2 = torch.zeros_like(freq)
        a0 = 1 + alpha / A
        a1 = -2.0 * torch.cos(omega)
        a2 = 1 - alpha / A

        a = torch.cat([a0, a1, a2], dim=1)
        b = torch.cat([b0, b1, b2], dim=1)

        b = b / a0
        a = a / a0
        return b, a