import torch
from glotnet.model.feedforward.wavenet import WaveNet
from glotnet.sigproc.levinson import spectrum_to_allpole
from glotnet.sigproc.lfilter import LFilter

def test_glotnet_shapes():

    timesteps = 1028
    batch = 1
    input_channels = 1
    output_channels = 1

    # STFT parameters
    n_fft = 512
    hop_length = 256
    win_length = 512
    p = 10

    wavenet = WaveNet(
        input_channels=2*input_channels,
        output_channels=output_channels,
        residual_channels=4,
        skip_channels=3,
        kernel_size=3,
        dilations=[1],
    )

    x = 0.1 * torch.randn(batch, input_channels, timesteps)

    X = torch.stft(x[:, 0, :], n_fft=n_fft,
                   hop_length=hop_length, win_length=win_length,
                   return_complex=True)
    # power spectrum
    X = torch.abs(X)**2

    # transpose to (batch, time, freq)
    X = X.transpose(1, 2)

    # allpole coefficients
    a, _ = spectrum_to_allpole(X, p)

    # transpose to (batch, order, num_frames)
    a = a.transpose(1, 2)

    lfilter = LFilter(n_fft=n_fft, hop_length=hop_length, win_length=win_length)
    # inverse filter
    e = lfilter.forward(x=x, b=a, a=None)

    x_prev = x[..., :-1]
    e_prev = e[..., :-1]

    # concatenate
    input = torch.cat((x_prev, e_prev), dim=1)

    params = wavenet.forward(input)