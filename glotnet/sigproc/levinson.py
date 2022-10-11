import torch
import torch.nn.functional as F

def toeplitz(r: torch.Tensor):
    """" Construct Toeplitz matrix """
    p = r.size(-1)   
    rr = torch.cat([r, r[..., 1:].flip(dims=(-1,))], dim=-1)
    T = [torch.roll(rr, i, dims=(-1,))[...,:p] for i in range(p)]
    return torch.stack(T, dim=-1)

def levinson(R, M):
    """ Levinson-Durbin method for converting autocorrelation to predictor polynomial
    Args:
        R: autocorrelation tensor, shape=(..., M) 
        M: filter polynomial order     
    Returns:
        A: filter predictor polynomial tensor, shape=(..., M)
    Note:
        R can contain more lags than M, but R[..., 0:M] are required 
    """
    E = R[..., 0:1]
    L = torch.cat([torch.ones_like(R[..., 0:1]),
                   torch.zeros_like(R[..., 0:M])], dim=-1)
    L_prev = L
    for p in torch.arange(0, M):
        K = torch.sum(L_prev[..., 0:p+1] * R[..., 1:p+2], dim=-1, keepdim=True) / E
        pad = torch.clamp(M-p-1, min=0)
        if p == 0:
            L = torch.cat([-1.0*K,
                           torch.ones_like(R[..., 0:1]),
                           torch.zeros_like(R[..., 0:pad])], dim=-1)
        else:
            L = torch.cat([-1.0*K,
                           L_prev[..., 0:p] - 1.0*K *
                           torch.flip(L_prev[..., 0:p], dims=[-1]),
                           torch.ones_like(R[..., 0:1]),
                           torch.zeros_like(R[..., 0:pad])], dim=-1)
        L_prev = L
        E = E * (1.0 - K ** 2)  # % order-p mean-square error
    L = torch.flip(L, dims=[-1])  # flip zero delay to zero:th index
    return L


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

    L = torch.cat([torch.ones_like(K[..., 0:1]), torch.zeros_like(K[..., 0:M])], dim=-1)
    L_prev = L 
    for p in torch.arange(0, M):
        pad = torch.clamp(M-p-1, min=0)
        if p == 0:
            L = torch.cat([-1.0*K[..., p:p+1],
                           torch.ones_like(K[..., 0:1]),
                           torch.zeros_like(K[..., 0:pad])], dim=-1)
        else:
            L = torch.cat([-1.0*K[..., p:p+1],
                           L_prev[..., 0:p] - 1.0*K[..., p:p+1] * torch.flip(L_prev[..., 0:p], dims=[-1]), # should be complex conjugate, if complex vals are used
                           torch.ones_like(K[..., 0:1]),
                           torch.zeros_like(K[..., 0:pad])], dim=-1)
        L_prev = L

    L = torch.flip(L, dims=[-1]) # flip zero delay to zero:th index
    return L