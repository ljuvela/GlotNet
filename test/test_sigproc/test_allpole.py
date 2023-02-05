import torch
from glotnet.sigproc.allpole import AllPole
from glotnet.sigproc.allpole import allpole
from torchaudio import functional as Fa

def test_allpole():

    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 1.0
    a = torch.tensor([1, -0.5, 0.2])
    T = x.shape[0]
    y = allpole(x=x.unsqueeze(0),
                a=a.reshape(1, -1, 1).expand(-1, -1, T))

    # test against torchaudio.functional.lfilter
    b = torch.zeros_like(a)
    b[0] = 1
    y2 = Fa.lfilter(waveform=x.unsqueeze(0),
                    b_coeffs=b.unsqueeze(0),
                    a_coeffs=a.unsqueeze(0),
                    clamp=False).squeeze(0)

    assert torch.allclose(y, y2)

def test_allpole_class():

    # TODO
    return 


    x = torch.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) * 1.0
    a = torch.tensor([1, -0.5, 0.2])

    T = x.shape[0]

    a_ref = a.clone().detach().requires_grad_(True)
    x_ref = x.clone().detach().requires_grad_(True)

    x.requires_grad_(True)
    a.requires_grad_(True)

    allpole = AllPole()

    y = allpole.forward(x=x.unsqueeze(0).unsqueeze(0),
                        a=a.reshape(1, -1, 1).expand(-1, -1, T))

    y.sum().backward()

    # test autograd against torchaudio.functional.lfilter
    b = torch.zeros_like(a_ref)
    b[0] = 1
    y_ref = Fa.lfilter(x_ref.unsqueeze(0), a_ref.unsqueeze(0), b.unsqueeze(0), clamp=False).squeeze(0)
    y_ref.sum().backward()

    assert torch.allclose(y, y_ref)

    assert x.grad is not None
    assert a.grad is not None

    print(a.grad)
    print(a_ref.grad)


    # check that gradients are the same
    assert torch.allclose(x.grad, x_ref.grad)
    assert torch.allclose(a.grad, a_ref.grad)



