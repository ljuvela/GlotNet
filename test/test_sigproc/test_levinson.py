import torch
from glotnet.sigproc.levinson import levinson, forward_levinson
from glotnet.sigproc.levinson import toeplitz

def test_levinson():

    # autocorrelation 
    p = 10
    r = torch.randn(p+1,)

    # solution by Toeplitz inverse
    R = toeplitz(r[:p])
    a_ref = torch.linalg.solve(R, r[1:])
    a_ref = torch.cat([torch.ones(1,), -1.0 * a_ref])

    # solution by Levinson-Durbin
    a = levinson(r, p)
 
    assert torch.allclose(a, a_ref, atol=1e-5, rtol=1e-4), \
        f'Results should match \n lev: {a} \n ref: {a_ref} '

