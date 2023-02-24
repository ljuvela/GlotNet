import torch
from glotnet.sigproc.levinson import spectrum_to_allpole

from glotnet.sigproc.lfilter import LFilter
from glotnet.sigproc.lpc import LinearPredictor
from glotnet.sigproc.oscillator import Oscillator


def test_inverse_filtering():

    # stft parameters
    n_fft = 512
    hop_length = 128
    win_length = 512
    fs = 16000

    batch = 1
    num_frames = 20

    # allpole order
    order = 10

    osc = Oscillator(audio_rate=fs, control_rate=fs//hop_length, shape='saw')
    x = osc.forward(f0=100 * torch.ones(batch, 1, num_frames))

    lpc = LinearPredictor(n_fft=n_fft, hop_length=hop_length, win_length=win_length, order=order)

    # allpole coefficients
    a = lpc.estimate(x=x[:, 0, :])

    # inverse filter
    e = lpc.inverse_filter(x=x, a=a)


def test_prediction():

    # stft parameters
    n_fft = 512
    hop_length = 128
    win_length = 512
    fs = 16000

    batch = 1
    num_frames = 20

    # allpole order
    order = 10

    osc = Oscillator(audio_rate=fs, control_rate=fs//hop_length, shape='saw')
    x = osc.forward(f0=100 * torch.ones(batch, 1, num_frames))


    lpc = LinearPredictor(n_fft=n_fft, hop_length=hop_length, win_length=win_length, order=order)
    a = lpc.estimate(x=x[:, 0, :])
    p = lpc.prediction(x=x, a=a)

    e = x - p

    # import matplotlib.pyplot as plt
    # plt.plot(p[0,0,:].detach().numpy())
    # plt.plot(x[0,0,:].detach().numpy())
    # plt.plot(e[0,0,:].detach().numpy())
    # plt.show()

def test_circular_consistency():

    # stft parameters
    n_fft = 512
    hop_length = 128
    win_length = 512
    fs = 16000

    batch = 1
    num_frames = 20

    # allpole order
    order = 10

    osc = Oscillator(audio_rate=fs, control_rate=fs//hop_length, shape='saw')
    x = osc.forward(f0=100 * torch.ones(batch, 1, num_frames))

    lpc = LinearPredictor(n_fft=n_fft, hop_length=hop_length, win_length=win_length, order=order)
    a = lpc.estimate(x=x[:, 0, :])
    p = lpc.prediction(x=x, a=a)
    e_inv = lpc.inverse_filter(x=x, a=a)

    e_diff = x - p

    # # # import ipdb; ipdb.set_trace()
    # import matplotlib.pyplot as plt
    # # plt.plot(p[0,0,:].detach().numpy())
    # plt.plot(e_diff[0,0,:].detach().numpy())
    # plt.plot(e_inv[0,0,:].detach().numpy())
    # # plt.plot(x[0,0,:].detach().numpy())
    # plt.show()

    assert torch.allclose(e_inv, e_diff, atol=1e-5, rtol=1e-4)
