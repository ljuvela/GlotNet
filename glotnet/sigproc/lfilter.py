import torch
import torch.nn.functional as F
import math

# TODO: wrap STFT into class
# TODO: support grouped convolution (multichannel, no mixing)

def ceil_division(n: int, d: int) -> int:
    """ Ceiling integer division """
    return -(n // -d)
class LFilter(torch.nn.Module):
    """ Linear filtering with STFT """

    def __init__(self, n_fft: int, hop_length: int, win_length: int):
        super().__init__()
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.win_length = win_length
        window = torch.hann_window(window_length=self.win_length)
        self.register_buffer('window', window.reshape(1, -1, 1), persistent=False)
        self.scale = 2.0 * self.hop_length / self.win_length

        # TODO: find scaling for synthesis window
        # print(f"window sqsum {self.window.square().sum()}")
        # print(f"window sum {self.window.sum()}")
        # print(f"hop {self.hop_length}, nfft {self.n_fft}, win {self.win_length}")
        # print(f"sum / sqsum : {self.window.sum()/self.window.square().sum()}")
        # print(f"sqsum / sum: {self.window.square().sum()/self.window.sum()}")

        self.fold_kwargs = {
            'kernel_size':(self.win_length, 1),
            'stride' : (self.hop_length, 1),
            'padding' : 0
        }

    def forward(self, x: torch.Tensor, b: torch.Tensor = None, a: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x : input signal
                (batch, channels, timesteps)
            b : filter numerator coefficients
                (batch, b_len, n_frames)
            a : filter denominator coefficients
                (batch, a_len, n_frames)
        """
        num_frames = ceil_division(x.size(-1), self.hop_length)

        left_pad = self.win_length
        last_frame_center = num_frames * self.hop_length
        right_pad = last_frame_center - x.size(-1) + self.win_length
        x_padded = F.pad(x, pad=(left_pad, right_pad))

        if x.size(1) != 1:
            raise RuntimeError("channels must be 1")
        # X = torch.stft(input=x_padded[:, 0, :],
        #                n_fft=self.n_fft,
        #                hop_length=self.hop_length,
        #                win_length=self.win_length,
        #                window=self.window,
        #                center=True,
        #                pad_mode='constant',
        #                onesided=True,
        #                return_complex=True)



        # frame
        x_padded = x_padded.unsqueeze(-1)
        fold_size = x_padded.shape[2:]
        x_framed = F.unfold(x_padded,  # (B, C, T, 1)
                            **self.fold_kwargs)

        # window
        x_windowed = x_framed * self.window

        # FFT
        X = torch.fft.rfft(x_windowed, n=self.n_fft, dim=1)

        def pad_frames(frames: torch.Tensor, target_len: int) -> torch.Tensor:
            n_pad = target_len - frames.size(-1)
            l_pad = n_pad // 2
            r_pad = ceil_division(n_pad, 2)
            return F.pad(frames, pad=(l_pad, r_pad), mode='replicate')

        if a is None:
            A = torch.ones_like(X)
        else:
            a_pad = pad_frames(a, target_len=X.size(-1))
            A = torch.fft.rfft(a_pad, n=self.n_fft, dim=1)
    
        if b is None:
            B = torch.ones_like(X)
        else:
            b_pad = pad_frames(b, target_len=X.size(-1))
            B = torch.fft.rfft(b_pad, n=self.n_fft, dim=1)

        # multiply
        Y = X * B / A

        # IFFT
        y_windowed = torch.fft.irfft(Y, n=self.n_fft, dim=1)
        y_windowed = y_windowed[:, :self.win_length, :]
        # TODO: window again for OLA
        # y_windowed = y_windowed[:, :self.win_length, :] * self.window
       

        # Overlap-add
        # TODO: fold does not work correctly if win length and fft length don't match!
        # TODO: change OLA fold args kernel size to fft_len
        y = F.fold(y_windowed, output_size=fold_size,
                   **self.fold_kwargs)
        return y[:, :, left_pad:-right_pad, 0] * self.scale

        # y = torch.istft(input=Y,
        #                n_fft=self.n_fft,
        #                hop_length=self.hop_length,
        #                win_length=self.win_length,
        #                window=self.window,
        #                center=True,
        #                onesided=True,
        #                )
        # y = y.unsqueeze(1)
        # return y[:, :, left_pad:-right_pad] 





