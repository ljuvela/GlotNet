import pytest
import torch
import torchaudio.functional as AF

from glotnet.sigproc import mulaw


@pytest.mark.parametrize("num_bits", [8, 10])
def test_mulaw_matches_torchaudio(num_bits):
    mu = mulaw.mu_from_bits(num_bits)
    x = torch.linspace(-1.0, 1.0, 10001)

    q = mulaw.encode(x, mu)
    q_ref = AF.mu_law_encoding(x, quantization_channels=mu + 1)
    assert torch.equal(q, q_ref)

    y = mulaw.decode(q, mu)
    y_ref = AF.mu_law_decoding(q_ref, quantization_channels=mu + 1)
    assert torch.allclose(y, y_ref, atol=1e-6)


def test_mulaw_companding_inverse():
    x = torch.linspace(-1.0, 1.0, 1001)
    y = mulaw.mu_law_encode(x)
    assert y.abs().max() <= 1.0
    assert torch.allclose(mulaw.mu_law_decode(y), x, atol=1e-6)


def test_mulaw_quantization_round_trip():
    mu = 255
    q = torch.arange(mu + 1)
    assert torch.equal(mulaw.encode(mulaw.decode(q, mu), mu), q)
    # endpoints and zero
    assert mulaw.encode(torch.tensor([-1.0, 1.0]), mu).tolist() == [0, mu]
    assert mulaw.decode(torch.tensor([0, mu]), mu).tolist() == [-1.0, 1.0]


def test_mulaw_clipping():
    q = mulaw.encode(torch.tensor([-3.0, 3.0]))
    assert q.tolist() == [0, 255]


def test_mulaw_quantization_error_small_near_zero():
    """ Companding: quantization step is finer near zero than near +-1 """
    x = torch.linspace(-1.0, 1.0, 20001)
    err = (mulaw.decode(mulaw.encode(x)) - x).abs()
    small = x.abs() < 0.01
    assert err[small].max() < 1e-3
    assert err.max() < 0.025  # half the largest step (~0.043 at |x| = 1)
