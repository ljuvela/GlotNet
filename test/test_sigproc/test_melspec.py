import torch
from glotnet.sigproc.melspec import LogMelSpectrogram

def test_allpole_from_log_spectrogram():

    melspec = LogMelSpectrogram()

    batch = 1
    time = 2048
    x = torch.randn(batch, 1, time)

    X = melspec(x)

    a = melspec.allpole(X)

    # import matplotlib.pyplot as plt

    # # import ipdb; ipdb.set_trace()
    # A = torch.fft.rfft(a, 512)

    # H = 20 * torch.log10(1/A[0,0, 5,:])
    # plt.plot(H)
    # plt.show()


