
import torch
from glotnet.sigproc.levinson import spectrum_to_allpole

from glotnet.sigproc.lfilter import LFilter
from glotnet.sigproc.oscillator import Oscillator


def test_inverse_filtering():

    # stft parameters
    n_fft = 512
    hop_length = 128
    win_length = 512
    fs = 16000

    batch = 1
    num_frames = 20

    # allpole order
    p = 10

    osc = Oscillator(audio_rate=fs, control_rate=fs//hop_length, shape='saw')
    x = osc.forward(f0=100 * torch.ones(batch, 1, num_frames))

    # x = torch.randn(1, 1, 2048)

    X = torch.stft(x[:, 0, :], n_fft=n_fft,
                   hop_length=hop_length, win_length=win_length,
                   return_complex=True)
    # power spectrum
    X = torch.abs(X)**2

    # transpose to (batch, time, freq)
    X = X.transpose(1, 2)

    # allpole coefficients
    a, _ = spectrum_to_allpole(X, p)

    # transpose to (batch, order, num_frames)
    a = a.transpose(1, 2)

    lfilter = LFilter(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # inverse filter
    e = lfilter.forward(x=x, b=a, a=None)

    # import matplotlib.pyplot as plt
    # plt.plot(e[0,0,:].detach().numpy())
    # plt.plot(x[0,0,:].detach().numpy())
    # plt.show()
