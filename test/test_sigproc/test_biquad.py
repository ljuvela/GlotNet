import torch
from importlib import reload

import glotnet.sigproc.biquad; reload(glotnet.sigproc.biquad)
from glotnet.sigproc.biquad import BiquadBandPassFunctional, BiquadBandPassModule
from glotnet.sigproc.oscillator import Oscillator

import pytest

def test_peak_biquad_gains():


    fs = 16000
    f0 = 100
    timesteps = 2048

    osc = Oscillator(audio_rate=fs, control_rate=fs)

    x_100 = osc(f0=100 * torch.ones(1, 1, timesteps))
    x_2000 = osc(f0=2000 * torch.ones(1, 1, timesteps))

    # gain in dB
    gain = 6.0
    # normalized frequency (nyquist = 1)
    freq = f0 / (fs / 2)
    biquad = BiquadBandPassModule(freq=freq, gain=gain)
    
    # pass throught biquad
    y_100 = biquad(x_100)
    y_2000 = biquad(x_2000)

    assert (y_100.max() - 10 ** (0.05 * gain)).abs() < 0.01
    assert (y_2000.max() - 1).abs() < 0.05


# TODO: test invalid normalized frequency (skip fs)

def test_peak_biquad_bank():
    fs = 16000
    batch = 1
    channels = 1
    timesteps = 2048

    # bank of biquads
    freq = torch.tensor([100.0, 1000.0, 2000.0])
    gain = torch.tensor([6.0, -6.0, 12.0])

    biquad = BiquadBandPassModule(freq=freq, gain=gain, fs=fs)

    x = torch.randn(batch, channels, timesteps)

    y = biquad.forward(x)
    

def test_biquad_backprop():

    pass


def test_peak_biquad_variable_freq():

    biquad = BiquadBandPassFunctional()





    