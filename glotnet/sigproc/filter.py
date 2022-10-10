import torch
import torch.nn.functional as F
import math

# TODO: wrap STFT into class

class LFilter(torch.nn.Module):
    """ Linear filtering with STFT """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        window = torch.hann_window(window_length=self.win_length)
        window = window.reshape(1, -1, 1)
        self.register_buffer('window', window, persistent=False)
        self.scale = 2.0 * self.hop_length / self.win_length


    def forward(self, x: torch.Tensor, b: torch.Tensor = None, a: torch.Tensor = None):
        """
        Args:
            x : input signal
                (batch, channels, timesteps)
            b : filter numerator coefficients
                (batch, b_len, n_frames)
            a : filter denominator coefficients
                (batch, a_len, n_frames)
        """
        num_frames = math.ceil(x.size(-1) / self.hop_length)

        left_pad = self.win_length
        last_frame_center = num_frames * self.hop_length
        right_pad = last_frame_center - x.size(-1) + self.win_length
        x_padded = F.pad(x, pad=(left_pad, right_pad)).unsqueeze(-1)

        print(f"pad left {left_pad}, right {right_pad}")

        fold_kwargs = {
            'kernel_size':(self.win_length, 1),
            'stride' : (self.hop_length, 1),
            'padding' : 0
        }
        # frame
        fold_size = x_padded.shape[2:]
        x_framed = F.unfold(x_padded,  # (B, C, T, 1)
                            **fold_kwargs)

        # window
        x_windowed = x_framed * self.window

        # FFT
        X = torch.fft.rfft(x_windowed, n=self.n_fft, dim=1)

        if a is None:
            A = torch.ones_like(X)
        else:
            A = torch.fft.rfft(a, n=self.n_fft, dim=1)
    
        if b is None:
            B = torch.ones_like(X)
        else:
            B = torch.fft.rfft(B, n=self.n_fft, dim=1)

        # multiply
        Y = X * B / A

        # IFFT
        y_windowed = torch.fft.irfft(Y, n=self.n_fft, dim=1)

        # Overlap-add
        y = F.fold(y_windowed, output_size=fold_size, 
        **fold_kwargs)

        return y[:, :, left_pad:-right_pad, 0] * self.scale




