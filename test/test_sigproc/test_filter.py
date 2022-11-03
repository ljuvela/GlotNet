import torch

# from importlib import reload
# import glotnet.sigproc.filter; reload(glotnet.sigproc.filter)

from glotnet.sigproc.lfilter import LFilter
from torchaudio.functional import lfilter as lfilter_ref
from glotnet.sigproc.lfilter import ceil_division

def test_lfilter_perferct_reconstruction():

    lfilter = LFilter(n_fft=16, hop_length=4, win_length=16)
    batch = 2
    channels = 1
    timesteps = 141
    x = torch.ones(batch, channels, timesteps)
    y = lfilter(x=x, b=None, a=None)

    assert torch.allclose(y, x)

def test_lfilter_causality():
    pass

# if __name__ == "__main__":
def test_lfilter_constant_filter_coefs():

    batch = 1
    channels = 1
    time = 1000
    x = 0.1 * torch.randn(batch, channels, time)

    ord = 1

    # a = torch.randn(20)
    a = torch.zeros(ord+1)
    a[0] = 1.0
    # a[1] = -0.8
    a = a / a[0]
    b = torch.randn(ord+1)

    b[0] = 1.0
    b[1] = torch.randn(1)
    y_ref = lfilter_ref(x, a_coeffs=a, b_coeffs=b, clamp=False)

    hop = 128
    num_frames = ceil_division(time, hop)
    lfilter = LFilter(n_fft=512, hop_length=hop, win_length=256)
    y = lfilter.forward(x=x,
                        b=b.reshape(1, -1, 1).expand(-1, -1, num_frames),
                        a=a.reshape(1, -1, 1).expand(-1, -1, num_frames))

    y = y.squeeze().detach()
    y_ref = y_ref.squeeze().detach()
    # assert torch.allclose(y_ref, y)
    err = (y_ref - y).squeeze().detach()

    # assert torch.allclose(y_ref / y_ref.norm(), y / y_ref.norm(), atol=1e-5, rtol=1e-4)

    norm = 1 / y_ref.norm()
    # close = torch.allclose(y_ref * norm, y * norm, atol=1e-3, rtol=1e-2)
    close = torch.allclose(y_ref * norm, y * norm, atol=1e-5, rtol=1e-4)
    # close = torch.allclose(y_ref * norm, y * norm, atol=1e-6, rtol=1e-5)
    assert(close)
    # print(close)
    # print(y.norm()/ y_ref.norm())

if __name__ == "__main__":
    test_lfilter_perferct_reconstruction()
    test_lfilter_constant_filter_coefs()