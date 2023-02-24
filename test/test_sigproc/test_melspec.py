import torch
from glotnet.sigproc.melspec import LogMelSpectrogram

def test_allpole_from_log_spectrogram():

    melspec = LogMelSpectrogram()

    batch = 1
    time = 2048
    x = torch.randn(batch, 1, time)

    X = melspec(x)

    a = melspec.allpole(X)


