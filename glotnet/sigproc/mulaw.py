""" Mu-law companding and quantization (ITU-T G.711)

    Signal values are in [-1, 1]. With `num_bits` bits, mu = 2**num_bits - 1
    and the quantized values are integers in [0, mu].

    Matches torchaudio.functional.mu_law_encoding / mu_law_decoding
    (with quantization_channels = mu + 1), without the torchaudio dependency.
"""
import math
import torch


def mu_from_bits(num_bits: int) -> int:
    """ Companding parameter mu = 2**num_bits - 1 (e.g. 255 for 8 bits) """
    if num_bits < 1:
        raise ValueError(f"num_bits must be positive, got {num_bits}")
    return 2 ** num_bits - 1


def mu_law_encode(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """ Compress: [-1, 1] -> [-1, 1]; input is clipped to [-1, 1] """
    x = x.clamp(-1.0, 1.0)
    return torch.sign(x) * torch.log1p(mu * x.abs()) / math.log1p(mu)


def mu_law_decode(y: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """ Expand: inverse of mu_law_encode """
    return torch.sign(y) * torch.expm1(y.abs() * math.log1p(mu)) / mu


def quantize(y: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """ Companded value in [-1, 1] -> class index in [0, mu] (int64) """
    q = torch.floor((y + 1.0) / 2.0 * mu + 0.5)
    return q.clamp(0, mu).to(torch.int64)


def dequantize(q: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """ Class index in [0, mu] -> companded value in [-1, 1] """
    return 2.0 * q.to(torch.get_default_dtype()) / mu - 1.0


def encode(x: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """ Signal -> class index (compress + quantize) """
    return quantize(mu_law_encode(x, mu), mu)


def decode(q: torch.Tensor, mu: int = 255) -> torch.Tensor:
    """ Class index -> signal (dequantize + expand) """
    return mu_law_decode(dequantize(q, mu), mu)
