import torch

from glotnet.sigproc.biquad import BiquadModule
from glotnet.sigproc.biquad import BiquadPeakFunctional
from glotnet.sigproc.biquad import BiquadResonatorFunctional
from glotnet.sigproc.biquad import BiquadBandpassFunctional
from glotnet.sigproc.biquad import BiquadParallelBankModule

from glotnet.sigproc.oscillator import Oscillator

import numpy as np

def test_peak_biquad_gains():


    fs = 16000
    f0 = 100
    timesteps = 2048

    osc = Oscillator(audio_rate=fs, control_rate=fs)

    x_100 = osc(f0=100 * torch.ones(1, 1, timesteps), init_phase=torch.pi)
    x_2000 = osc(f0=2000 * torch.ones(1, 1, timesteps), init_phase=torch.pi)

    # gain in dB
    gain = 6.0
    # normalized frequency (nyquist = 1)
    freq = f0 / (fs / 2)
    biquad = BiquadModule(func=BiquadPeakFunctional())
    biquad.set_freq(freq)
    biquad.set_gain_dB(gain)

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
    freq = torch.tensor([100.0, 1000.0, 2000.0, 3000.0])
    gain = torch.tensor([6.0, -6.0, 12.0, 0.0])

    biquad = BiquadModule(in_channels=1, out_channels=4, fs=fs, func=BiquadPeakFunctional())
    biquad.set_freq(freq.reshape(-1, 1))
    biquad.set_gain_dB(gain.reshape(-1, 1))

    osc = Oscillator(audio_rate=fs, control_rate=fs)    

    x = osc(f0=freq.reshape(1, -1, 1) * torch.ones(batch, channels, timesteps), init_phase=torch.pi)

    # TODO: implementation with groups

    y = torch.zeros_like(x)
    for i, xi in enumerate(x.chunk(x.shape[1], dim=1)):
        yi = biquad.forward(xi)
        y[:, i, :] = yi[:, i, :]

    for i, _ in enumerate(freq):
        assert (y[:, i, :].max() - 10 ** (0.05 * gain[i])).abs() < 0.01


def test_biquad_backprop():

    pass


def test_peak_biquad_variable_freq():

    biquad = BiquadPeakFunctional()

def test_resonator_biquad():

    # get a resonator impulse response
    fs = 16000
    f0 = 3000
    timesteps = 2048
    nfft=2048
    fbins = nfft // 2 + 1

    biquad = BiquadModule(fs=fs, func=BiquadResonatorFunctional())
    biquad.set_freq(f0)
    biquad.set_gain_dB(0.0)
    biquad.set_Q(10.0)

    h = biquad.get_impulse_response(n_timesteps=timesteps)

    H = biquad.get_frequency_response(n_timesteps=timesteps, n_fft=timesteps)

    assert H.argmax() == f0 * nfft / fs

    # import matplotlib.pyplot as plt
    # h = h.squeeze().detach().numpy()
    # H = H.squeeze().detach().numpy()
    # f = np.linspace(0, fs / 2, fbins)
    # plt.close('all')
    # plt.figure()
    # plt.plot(h)

    # plt.figure()
    # plt.plot(f, 20 * np.log10(H))

    # plt.figure()
    # plt.semilogx(f, 20 * np.log10(H))
    # plt.show()


def test_biquad_parallel_bank():

    biquad = BiquadParallelBankModule(num_filters=4, func=BiquadResonatorFunctional())
    biquad.filter_bank.set_Q(torch.tensor([10.0, 10.0, 10.0, 10.0]))


    n_fft = 2048
    fbins = n_fft // 2 + 1
    fs = 2


    # import ipdb; ipdb.set_trace()
    h = biquad.get_impulse_response(n_timesteps=2048)

    H = biquad.get_frequency_response(n_timesteps=2048, n_fft=2048)

    # import matplotlib.pyplot as plt
    # H = H.squeeze().detach().numpy()
    # f = np.linspace(0, fs / 2, fbins)
    # plt.close('all')

    # plt.figure()
    # plt.plot(f, 20 * np.log10(H))

    # plt.figure()
    # plt.semilogx(f, 20 * np.log10(H))
    # plt.show()



def test_biquad_parallel_bank2():


    n_fft = 2048
    fbins = n_fft // 2 + 1
    fs = 16000

    # biquad = BiquadParallelBankModule(num_filters=2, func=BiquadPeakFunctional())
    # biquad.set_gain_dB(torch.tensor([6.0, -6.0]))
    # biquad.set_freq(torch.tensor([0.1, 0.9]))
    # # biquad.set_Q(torch.tensor([2.0, 2.0]))


    # biquad = BiquadParallelBankModule(num_filters=1, func=BiquadPeakFunctional())
    # biquad.set_gain_dB(torch.tensor([2.0]))
    # biquad.set_freq(torch.tensor([0.1,]))
    # biquad.set_Q(torch.tensor([2.0]))

    # biquad = BiquadParallelBankModule(num_filters=2, func=BiquadPeakFunctional())
    # biquad.set_gain_dB(torch.tensor([0.0, 0.0]))
    # biquad.set_freq(torch.tensor([0.1, 0.9]))
    # biquad.set_Q(torch.tensor([2.0, 2.0]))




    biquad = BiquadParallelBankModule(
        num_filters=2,
        fs=fs,
        func=BiquadBandpassFunctional())
    biquad.set_gain_dB(torch.tensor([6.0, -6.0]))
    biquad.set_freq(torch.tensor([100, 1000]))
    biquad.set_Q(torch.tensor([2.0, 2.0]))


    # biquad = BiquadParallelBankModule(
    #     num_filters=2,
    #     fs=fs,
    #     func=BiquadBandpassFunctional())
    # # biquad.set_gain_dB(torch.tensor([6.0, -6.0]))
    # biquad.set_freq(torch.tensor([100, 1000]))
    # biquad.set_Q(torch.tensor([2.0, 2.0]))

    h = biquad.get_impulse_response(n_timesteps=2048)

    H = biquad.get_frequency_response(n_timesteps=2048, n_fft=2048)

    # import matplotlib.pyplot as plt
    # H = H.squeeze().detach().numpy()
    # f = np.linspace(0, fs / 2, fbins)
    # plt.close('all')

    # plt.figure()
    # plt.plot(f, 20 * np.log10(H))

    # plt.figure()
    # plt.semilogx(f, 20 * np.log10(H))
    # plt.show()