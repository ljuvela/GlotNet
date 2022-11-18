import torch
import torchaudio.functional as F

def allpole(x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """
    All-pole filter
    :param x: input signal, shape (..., T)
    :param a: filter coefficients (denominator), shape (..., p)
    :return: filtered signal
    """
    y = torch.zeros_like(x)

    a_normalized = a / a[..., 0:1]

    # filter order
    p = a.shape[-1] - 1

    # filter coefficients
    a1 = a_normalized[..., 1:]

    # flip coefficients
    a1 = torch.flip(a1, [-1])

    # zero pad y with filter order
    y = torch.nn.functional.pad(y, (p, 0))

    # filter
    for i in range(p, y.shape[-1]):
        y[..., i] = x[..., i - p] - torch.sum(a1 * y[..., i-p:i], dim=-1)

    return y[..., p:]



def test_allpole():
    # test against scipy.signal.lfilter


    import scipy.signal
    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 1.0
    a = torch.tensor([1, -0.5, 0.2])
    # a = torch.tensor([1, -0.5])
    y = allpole(x, a)

    # test against torchaudio.functional.lfilter
    b = torch.zeros_like(a)
    b[0] = 1
    y2 = F.lfilter(x.unsqueeze(0), a.unsqueeze(0), b.unsqueeze(0), clamp=False).squeeze(0)

    assert torch.allclose(y, y2)

if __name__ == "__main__":
    test_allpole()