import torch
from glotnet.model.feedforward.wavenet import WaveNet
from glotnet.losses.distributions import GaussianDensity

from glotnet.sigproc.levinson import spectrum_to_allpole
from glotnet.sigproc.lfilter import LFilter

from glotnet.model.autoregressive.glotnet import GlotNetAR

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


def test_glotnet_ar_order_zero():

    order = 1

    dist = GaussianDensity()
    model = GlotNetAR(input_channels=3, output_channels=2,
                      residual_channels=2, skip_channels=2,
                      kernel_size=2, dilations=[1], distribution=dist,
                      hop_length=1, lpc_order=order)

    batch = 1
    channels = 1
    timesteps = 3
    x = torch.zeros(batch, channels, timesteps)
    a = torch.zeros(batch, order + 1, timesteps)
    a[:, 0, :] = 1.0
    a[:, 1, :] = 0

    temp = torch.zeros(batch, channels, timesteps)
    y_ref = model.forward(input=x, a=a, temperature=temp)
    y_ext = model.inference(input=x, a=a, temperature=temp)

    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"
    


def test_glotnet_ar_order_one():

    order = 1

    dist = GaussianDensity(temperature=0.0)
    model = GlotNetAR(input_channels=3, output_channels=2,
                      residual_channels=2, skip_channels=2,
                      kernel_size=2, dilations=[1], distribution=dist,
                      hop_length=1, lpc_order=order)

    batch = 1
    channels = 1
    timesteps = 3
    x = torch.zeros(batch, channels, timesteps)
    a = torch.zeros(batch, order + 1, timesteps)
    a[:, 0, :] = 1.0
    a[:, 1, :] = -0.9

    temp = torch.zeros(batch, channels, timesteps)
    y_ref = model.forward(input=x, a=a, temperature=temp)
    y_ext = model.inference(input=x, a=a, temperature=temp)


    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"
    

def test_glotnet_ar_order_two():

    order = 2

    dist = GaussianDensity(temperature=0.0)
    model = GlotNetAR(input_channels=3, output_channels=2,
                      residual_channels=2, skip_channels=2,
                      kernel_size=2, dilations=[1], distribution=dist,
                      hop_length=1, lpc_order=order)

    batch = 1
    channels = 1
    timesteps = 3
    x = torch.zeros(batch, channels, timesteps)
    a = torch.zeros(batch, order + 1, timesteps)
    a[:, 0, :] = 1.0
    a[:, 1, :] = -0.5
    a[:, 2, :] = 0.2

    temp = torch.zeros(batch, channels, timesteps)
    y_ref = model.forward(input=x, a=a, temperature=temp)
    y_ext = model.inference(input=x, a=a, temperature=temp)


    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"


# test with frame-based a


def test_glotnet_ar_framed():

    order = 2
    hop_length = 5

    dist = GaussianDensity(temperature=0.0)
    model = GlotNetAR(input_channels=3, output_channels=2,
                      residual_channels=2, skip_channels=2,
                      kernel_size=2, dilations=[1], distribution=dist,
                      hop_length=hop_length, lpc_order=order)

    batch = 1
    channels = 1
    timesteps = 10
    x = torch.zeros(batch, channels, timesteps)
    a = torch.zeros(batch, order + 1, timesteps//hop_length)
    a[:, 0, :] = 1.0
    a[:, 1, :] = -0.5
    a[:, 2, :] = 0.2

    temp = torch.zeros(batch, channels, timesteps)
    y_ref = model.forward(input=x, a=a, temperature=temp)
    y_ext = model.inference(input=x, a=a, temperature=temp)

    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"


def test_glotnet_cond():

    torch.manual_seed(42)

    order = 2
    hop_length = 5
    cond_channels = 1

    dist = GaussianDensity(temperature=0.0)
    model = GlotNetAR(input_channels=3, output_channels=2,
                      residual_channels=2, skip_channels=2,
                      kernel_size=2, dilations=[1, 2],
                      cond_channels=cond_channels,
                      distribution=dist,
                      hop_length=hop_length, lpc_order=order)

    batch = 1
    channels = 1
    timesteps = 10
    x = torch.zeros(batch, channels, timesteps)
    a = torch.zeros(batch, order + 1, timesteps//hop_length)
    a[:, 0, :] = 1.0
    a[:, 1, :] = -0.5
    a[:, 2, :] = 0.2
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    temp = torch.zeros(batch, channels, timesteps)
    y_ref = model.forward(input=x, a=a, cond_input=c, temperature=temp)
    y_ext = model.inference(input=x, a=a, cond_input=c, temperature=temp)

    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"



def test_glotnet_cond_sample_after_filtering():

    torch.manual_seed(42)

    order = 2
    hop_length = 5
    cond_channels = 1

    dist = GaussianDensity(temperature=0.0)
    model = GlotNetAR(input_channels=3, output_channels=2,
                      residual_channels=2, skip_channels=2,
                      kernel_size=2, dilations=[1, 2],
                      cond_channels=cond_channels,
                      distribution=dist,
                      hop_length=hop_length,
                      lpc_order=order, 
                      sample_after_filtering=True)

    batch = 1
    channels = 1
    timesteps = 10
    x = torch.zeros(batch, channels, timesteps)
    a = torch.zeros(batch, order + 1, timesteps//hop_length)
    a[:, 0, :] = 1.0
    a[:, 1, :] = -0.5
    a[:, 2, :] = 0.2
    c = 0.1 * torch.randn(batch, cond_channels, timesteps)

    temp = torch.zeros(batch, channels, timesteps)
    y_ref = model.forward(input=x, a=a, cond_input=c, temperature=temp)
    y_ext = model.inference(input=x, a=a, cond_input=c, temperature=temp)

    assert torch.allclose(y_ref, y_ext, atol=1e-5, rtol=1e-5), \
        f"Outputs must match \n ref: {y_ref} \n ext: {y_ext}"
