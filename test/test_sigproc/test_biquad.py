import torch
from importlib import reload

import glotnet.sigproc.biquad; reload(glotnet.sigproc.biquad)
from glotnet.sigproc.biquad import BiquadBandPassFunctional, BiquadBandPassModule
from glotnet.sigproc.oscillator import Oscillator


def test_peak_biquad_gains():


    fs = 16000
    f0 = 100

    osc = Oscillator(audio_rate=fs, control_rate=fs)

    x_100 = osc(f0=100 * torch.ones(1, 1, 2048))
    x_2000 = osc(f0=2000 * torch.ones(1, 1, 2048))

    # gain in dB
    gain = 6.0
    # normalized frequency (nyquist = 1)
    freq = f0 / (fs / 2)
    biquad = BiquadBandPassModule(freq=freq, gain=gain)
    
    y_100 = biquad(x_100)
    y_2000 = biquad(x_2000)

    assert (y_100.max() - 10 ** (0.05 * gain)).abs() < 0.01
    assert (y_2000.max() - 1).abs() < 0.05