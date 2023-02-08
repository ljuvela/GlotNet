import torch

from glotnet.sigproc.emphasis import Emphasis
from glotnet.sigproc.oscillator import Oscillator

from torchaudio.functional import lfilter

def test_emphasis():

    batch = 1
    time = 2048
    osc = Oscillator(audio_rate=16000, control_rate=16000)
    x = osc.forward(1000*torch.ones(batch, 1, time))

    emphasis = Emphasis(alpha=0.95)

    y = emphasis(x)

    # torchaudio lfilter reference
    y_ref = lfilter(x, b_coeffs=torch.tensor([1, -0.95]), a_coeffs=torch.tensor([1, 0]), clamp=False)

    norm = 1 / y_ref.norm()
    assert torch.allclose(y * norm, y_ref * norm, atol=1e-5, rtol=1e-4), \
        f"y: {y} y_ref: {y_ref}"


def test_deemphasis():

    batch = 1
    time = 2048
    osc = Oscillator(audio_rate=16000, control_rate=16000)
    x = osc.forward(1000*torch.ones(batch, 1, time))

    emphasis = Emphasis(alpha=0.95)

    y = emphasis.deemphasis(x)

    # torchaudio lfilter reference
    y_ref = lfilter(x, a_coeffs=torch.tensor([1, -0.95]), b_coeffs=torch.tensor([1, 0]), clamp=False)

    norm = 1 / y_ref.norm()
    assert torch.allclose(y * norm, y_ref * norm, atol=1e-4, rtol=1e-3), \
        f"y: {y} y_ref: {y_ref}"



