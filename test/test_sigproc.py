import torch
from glotnet.sigproc.filter import LFilter

def test_lfilter_perferct_reconstruction():

    lfilter = LFilter(n_fft=16, hop_length=4, win_length=16)
    batch = 4
    channels = 1
    timesteps = 141
    x = torch.ones(batch, channels, timesteps)

    y = lfilter(x=x, b=None, a=None)

    assert torch.allclose(y, torch.ones_like(y))

def test_lfilter_causality():
    pass

if __name__ == "__main__":
    test_lfilter_perferct_reconstruction()