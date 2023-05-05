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
        Q = Q.reshape(-1, 1, Q.size(-1))

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


    @property 
    def freq(self):
        return self._freq
    
    @freq.setter
    def freq(self, freq):
        if type(freq) != torch.Tensor:
            freq = torch.tensor([freq], dtype=torch.float32)

        # convert to normalized frequency
        freq = 2.0 * freq / self.fs

        if freq.max() > 1.0:
            raise ValueError(
                "Maximum normalized frequency is larger than 1.0. "
                "Please provide a sample rate or input normalized frequencies")
        if freq.min() < 0.0:
            raise ValueError(
                "Maximum normalized frequency is smaller than 0.0.")


        self._freq.data = freq * torch.ones_like(self._freq)

    @property 
    def gain_dB(self):
        return self._gain_dB

    @gain_dB.setter
    def gain_dB(self, gain):
        if type(gain) != torch.Tensor:
            gain = torch.tensor([gain], dtype=torch.float32)
        self._gain_dB.data = gain * torch.ones_like(self._gain_dB)

    @property
    def Q(self):
        return self._Q
    
    @Q.setter
    def Q(self, Q):
        if type(Q) != torch.Tensor:
            Q = torch.tensor([Q], dtype=torch.float32)
        self._Q.data = Q * torch.ones_like(self._Q)

    def _init_freq(self):
        freq = torch.rand(1, self.channels_in, self.channels_out)
        self._freq = torch.nn.Parameter(freq)

    def _init_gain_dB(self):
        gain_dB = torch.zeros(1, self.channels_in, self.channels_out)
        self._gain_dB = torch.nn.Parameter(gain_dB)

    def _init_Q(self):
        Q = torch.ones(1, self.channels_in, self.channels_out)
        self._Q = torch.nn.Parameter(0.7071 * Q)



    def __init__(self,
                 channels_in: int=1,
                 channels_out: int=1,
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

        self.channels_in = channels_in
        self.channels_out = channels_out

        self._init_freq()
        self._init_gain_dB()
        self._init_Q()


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

        batch, channels, timesteps = x.size()

        # Expand time dimension
        # TODO: separate backend for time-constant params
        timesteps = x.size(2)
        num_frames = timesteps // self.func.hop_length
        # size is (batch, channels_in, channels_out, n_frames)
        freq = freq.unsqueeze(-1).expand(-1, -1, -1, num_frames)
        gain = gain.unsqueeze(-1).expand(-1, -1, -1, num_frames)
        Q = Q.unsqueeze(-1).expand(-1, -1, -1, num_frames)

        # reshape parameters
        # freq = freq.reshape(-1, 1, num_frames)
        # gain = gain.reshape(-1, 1, num_frames)
        # Q = Q.reshape(-1, 1, num_frames)

        y = self.func.forward(x, freq, gain, Q)

        # reshape outputs
        # y = y.reshape(batch, channels, -1)


        return y

class BiquadParallelBankModule(torch.nn.Module):

    def __init__(self, 
                 num_filters:int=10, 
                 func: BiquadBaseFunctional = BiquadPeakFunctional()
                ):
        """
        Args:
            num_filters: number of filters in bank
            func: BiquadBaseFunctional subclass

        """
        super().__init__()

        self.num_filters = num_filters
        # flat initialization
        freq = torch.linspace(0.0, 1.0, num_filters+2)[1:-1]
        gain = torch.zeros_like(freq)
        Q = 0.7071 * torch.ones_like(freq)

        self.filter_bank = BiquadModule(channels_in=1, channels_out=num_filters, func=func)
        self.filter_bank.freq = freq
        self.filter_bank.gain_dB = gain
        self.filter_bank.Q = Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x, shape is (batch, channels=1, timesteps)

        Returns:
            y, shape is (batch, channels=1, timesteps)
        """

        if x.size(1) != 1:
            raise ValueError(f"Input must have 1 channel, got {x.size(1)}")

        # expand channels to match filter bank
        # x = x.expand(-1, self.num_filters, -1)

        import ipdb; ipdb.set_trace()

        # output shape is (batch, channels, , timesteps)
        y = self.filter_bank(x)

        # parallel filters are summed
        y = y.sum(dim=-2, keepdim=False)
        return y

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