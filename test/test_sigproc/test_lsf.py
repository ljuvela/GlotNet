import torch
from glotnet.sigproc.lsf import conv, lsf2poly


def test_conv():

    x = torch.tensor([1.0, 2.0, 3.0]).reshape(1,1,-1)
    y = torch.tensor([1.0, 2.0, 3.0]).reshape(1,1,-1)

    z = conv(x, y)

    assert torch.allclose(z, torch.tensor([1.0, 4.0, 10.0, 12.0, 9.0]))


def test_lsf_flat():

    order = 8
    lsf = torch.pi * torch.linspace(0.0, 1.0, order+2)[1:-1].reshape(1, 1, -1)
    a_ref = torch.zeros(1, 1, order+1)
    a_ref[..., 0] = 1.0
    a = lsf2poly(lsf)
    assert torch.allclose(a, a_ref, atol=1e-5, rtol=1e-4), f"{a} != {a_ref}"

def test_lsf():

    lsf = torch.tensor([0.7842 , 1.5605 , 1.8776 , 1.8984, 2.3593]).reshape(1, 1, -1)
    
    a_ref = torch.tensor([ 1.00000000e+00, 6.14837835e-01, 9.89884967e-01, 9.31594056e-05, 3.13713832e-03, -8.12002261e-03])

    a = lsf2poly(lsf)

    assert torch.allclose(a, a_ref, atol=1e-5, rtol=1e-4), f"{a} != {a_ref}"