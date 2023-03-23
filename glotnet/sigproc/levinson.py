import torch
import torch.nn.functional as F

def toeplitz(r: torch.Tensor):
    """" Construct Toeplitz matrix """
    p = r.size(-1)   
    rr = torch.cat([r, r[..., 1:].flip(dims=(-1,))], dim=-1)
    T = [torch.roll(rr, i, dims=(-1,))[...,:p] for i in range(p)]
    return torch.stack(T, dim=-1)

def levinson(R, M, eps=1e-3):
    """ Levinson-Durbin method for converting autocorrelation to predictor polynomial
    Args:
        R: autocorrelation tensor, shape=(..., M) 
        M: filter polynomial order     
    Returns:
        A: filter predictor polynomial tensor, shape=(..., M)
    Note:
        R can contain more lags than M, but R[..., 0:M] are required 
    """
    # normalize R
    R = R / R[..., 0:1]
    # white noise correction
    R[..., 0] = R[..., 0] + eps

    E = R[..., 0:1]
    A = torch.cat([torch.ones_like(R[..., 0:1]),
                   torch.zeros_like(R[..., 0:M])], dim=-1)
    A_prev = A
    for p in torch.arange(0, M):
        K = torch.sum(A_prev[..., 0:p+1] * R[..., 1:p+2], dim=-1, keepdim=True) / E
        if K.abs().max() > 1.0:
            raise ValueError(f"Unstable filter, |K| was {K.abs().max()}")
        pad = torch.clamp(M-p-1, min=0)
        if p == 0:
            A = torch.cat([-1.0*K,
                           torch.ones_like(R[..., 0:1]),
                           torch.zeros_like(R[..., 0:pad])], dim=-1)
        else:
            A = torch.cat([-1.0*K,
                           A_prev[..., 0:p] - 1.0*K *
                           torch.flip(A_prev[..., 0:p], dims=[-1]),
                           torch.ones_like(R[..., 0:1]),
                           torch.zeros_like(R[..., 0:pad])], dim=-1)
        A_prev = A
        E = E * (1.0 - K ** 2)  # % order-p mean-square error
    A = torch.flip(A, dims=[-1])  # flip zero delay to zero:th index
    return A


def forward_levinson(K, M=None):
    """ Forward Levinson method for converting reflection coefficients to direct form polynomial

        Args:
            K: reflection coefficient tensor, shape=(..., M) 
            M: filter polynomial order (optional)
        Returns:
            A: filter predictor polynomial tensor, shape=(..., M)
        Note:
            K can contain more stages than M, but K[..., 0:M] are required 

    """
    if M is None:
        M = K.size(-1)

    A = -1.0*K[..., 0:1]
    for p in torch.arange(1, M):
        A = torch.cat([-1.0*K[..., p:p+1],
                        A[..., 0:p] - 1.0*K[..., p:p+1] * torch.flip(A[..., 0:p], dims=[-1])], dim=-1)

    A = torch.cat([A, torch.ones_like(A[..., 0:1])], dim=-1)
    A = torch.flip(A, dims=[-1]) # flip zero delay to zero:th index


    return A





def spectrum_to_allpole(spectrum:torch.Tensor, order:int, root_scale:float=1.0):
    """ Convert spectrum to all-pole filter coefficients
    
    Args:
        spectrum: power spectrum (squared magnitude), shape=(..., K)
        order: filter polynomial order

    Returns:
        g: filter gain
        a: filter predictor polynomial tensor, shape=(..., order+1)
    """
    r = torch.fft.irfft(spectrum, dim=-1)
    # add small value to diagonal to avoid singular matrix
    r[..., 0] = r[..., 0] + 1e-6 
    # all pole from autocorr
    a = levinson(r, order)

    # filter gain
    # g = torch.sqrt(torch.dot(r[:(order+1)], a))
    g = torch.sqrt(torch.sum(r[..., :(order+1)] * a, dim=-1, keepdim=True))

    # scale filter roots
    if root_scale < 1.0:
        a = a * root_scale ** torch.arange(order+1, dtype=torch.float32, device=a.device)

    return a, g