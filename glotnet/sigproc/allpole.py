import torch
import torchaudio.functional as Fa
import torch.nn.functional as F

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

class AllPoleFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, a):
        y = allpole(x, a)
        ctx.save_for_backward(y, x, a)
        return y

    @staticmethod
    def backward(ctx, dy):
        y, x, a = ctx.saved_tensors
        dx = da = None

        n_batch = x.size(0)
        n_channels = x.size(1)
        n_order = a.size(-1) - 1

        # filter or
        dyda = allpole(-y, a)
        dyda = torch.nn.functional.pad(dyda, (n_order, 0))

        if a.requires_grad:
            da = F.conv1d(
                dyda.view(1, n_batch * n_channels, -1),
                dy.view(n_batch * n_channels, 1, -1),
                groups=n_batch * n_channels).view(n_batch, n_channels, -1).sum(0).flip(1)
            
        if x.requires_grad:
            dx = allpole(dy.flip(-1), a).flip(-1)

        return dx, da

class AllPole(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x, a):
        return AllPoleFunction.apply(x, a)

def test_allpole_class():

    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 1.0
    a = torch.tensor([1, -0.5, 0.2])

    a_ref = a.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)

    x.requires_grad_(True)
    a.requires_grad_(True)

    allpole = AllPole()

    y = allpole(x.unsqueeze(0).unsqueeze(0),  a.unsqueeze(0))
    y.sum().backward()

    # test autograd against torchaudio.functional.lfilter
    b = torch.zeros_like(a_ref)
    b[0] = 1
    y_ref = Fa.lfilter(x_ref.unsqueeze(0), a_ref.unsqueeze(0), b.unsqueeze(0), clamp=False).squeeze(0)
    y_ref.sum().backward()

    assert torch.allclose(y, y_ref)

    assert x.grad is not None
    assert a.grad is not None

    # check that gradients are the same
    assert torch.allclose(x.grad, x_ref.grad)
    assert torch.allclose(a.grad, a_ref.grad)


def test_allpole():

    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 1.0
    a = torch.tensor([1, -0.5, 0.2])
    # a = torch.tensor([1, -0.5])
    y = allpole(x, a)

    # test against torchaudio.functional.lfilter
    b = torch.zeros_like(a)
    b[0] = 1
    y2 = Fa.lfilter(x.unsqueeze(0), a.unsqueeze(0), b.unsqueeze(0), clamp=False).squeeze(0)

    assert torch.allclose(y, y2)

if __name__ == "__main__":
    test_allpole()
    test_allpole_class()