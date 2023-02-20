import torch
from glotnet.sigproc.levinson import levinson, forward_levinson
from glotnet.sigproc.levinson import toeplitz
from glotnet.sigproc.levinson import spectrum_to_allpole

def test_levinson():

    # autocorrelation 
    p = 10
    r = 0.1 * torch.randn(p+1,)
    r[0] = 1.0

    # solution by Toeplitz inverse
    R = toeplitz(r[:p])
    a_ref = torch.linalg.solve(R, r[1:])
    a_ref = torch.cat([torch.ones(1,), -1.0 * a_ref])

    # solution by Levinson-Durbin
    a = levinson(r, p, eps=1e-9)
 
    assert torch.allclose(a, a_ref, atol=1e-5, rtol=1e-4), \
        f'Results should match \n lev: {a} \n ref: {a_ref} '


def test_spectrum_to_allpole():

    # filter order
    p = 10

    # frequency bins
    n_fft = 512
    fbins = n_fft // 2 + 1

    # flat spectrum
    H = torch.ones(fbins,)

    # convert to all-pole
    a, g = spectrum_to_allpole(H, p)

    # get spectrum from all-pole
    A = torch.fft.rfft(a, n_fft)
    H2 = torch.abs(g / A)

    # check that all-pole is close to an impulse
    assert torch.allclose(a, torch.cat([torch.ones(1,), torch.zeros(p,)]), atol=1e-5, rtol=1e-4), \
        f'Results should match \n lev: {a} \n ref: {torch.cat([torch.ones(1,), torch.zeros(p,)])} '

    # check that spectrum is close to flat
    assert torch.allclose(H2, H, atol=1e-5, rtol=1e-4), \
        f'Results should match \n lev: {H2} \n ref: {H} '



# def test_spectrum_to_allpole_bct():

#     # filter order
#     p = 10

#     # frequency bins
#     n_fft = 512
#     fbins = n_fft // 2 + 1

#     # flat spectrum
#     batch = 3
#     channels = 2
#     H = torch.ones(batch, channels, fbins)

#     # convert to all-pole
#     g, a = spectrum_to_allpole(H, p)

#     # get spectrum from all-pole
#     A = torch.fft.rfft(a, n_fft, dim=-1)
#     H2 = torch.abs(g / A)

#     # check that all-pole is close to an impulse
#     assert torch.allclose(a, torch.cat([torch.ones(1,), torch.zeros(p,)]), atol=1e-5, rtol=1e-4), \
#         f'Results should match \n lev: {a} \n ref: {torch.cat([torch.ones(1,), torch.zeros(p,)])} '

#     # check that spectrum is close to flat
#     assert torch.allclose(H2, H, atol=1e-5, rtol=1e-4), \
#         f'Results should match \n lev: {H2} \n ref: {H} '
