import torch
from glotnet.model.feedforward.wavenet import WaveNet
from glotnet.model.autoregressive.wavenet import WaveNetAR
from glotnet.losses.distributions import GaussianDensity
import numpy as np

# def train():
if __name__ == "__main__":

    # data
    f0 = 100
    fs = 16000
    dur = 1
    t = torch.linspace(0, dur * fs, dur * fs) / fs
    twopi = 2 * torch.pi
    phi = twopi * torch.rand(1)
    x = torch.sin(twopi * f0 * t + phi)
    x = x.unsqueeze(0).unsqueeze(0)

    R = 8
    S = 8
    D = [1, 2, 4, 8, 16]

    # model
    model = WaveNet(input_channels=1, output_channels=2, residual_channels=R,
                    skip_channels=S, kernel_size=3, dilations=D)

    model_ar = WaveNetAR(input_channels=1, output_channels=2, residual_channels=R,
                    skip_channels=S, kernel_size=3, dilations=D,
                    distribution=GaussianDensity(temperature=1.0))

    # method 1
    # for p_src, p_dst in zip(model.parameters(), model_ar.parameters()):
    #     p_dst.data = p_src.data
    # method 2 
    model_ar.load_state_dict(model.state_dict(), strict=False)
    ar_input = torch.zeros(1, 1, 16000)
    x_gen_init = model_ar.forward(input=ar_input)


    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    # loss
    criterion = GaussianDensity()

    # iterate
    max_iters = 2000
    for iter in range(max_iters):

        x_curr = x[:, :, 1:]
        x_prev = x[:, :, :-1]

        params = model(x_prev)

        nll = criterion.nll(x=x_curr, params=params)
        loss = nll.mean()
        loss.backward()

        m = nll

        if iter % 100 == 0:
            print(f"Loss {iter}: {loss}")

        optim.step()
        optim.zero_grad()

    print("Finished training")

    m = params[:, 0:1, :]
    log_sig = params[:, 1:2, :]

    model_ar.load_state_dict(model.state_dict(), strict=False)
    x_gen_final = model_ar.forward(input=ar_input)

    model_ar.distribution.set_temperature(0.1)
    x_gen_cold = model_ar.forward(input=ar_input)

    from matplotlib import pyplot as plt

    plt.figure(1); plt.clf()
    plt.plot(x_curr.squeeze())
    plt.plot(m.squeeze().detach())
    plt.legend(['ref', 'model mean'])

    plt.figure(2); plt.clf()
    plt.plot(x_gen_final[:,0,:].squeeze().detach())
    plt.plot(x_gen_cold[:,0,:].squeeze().detach())
    plt.title("generated signal")
    plt.legend(['temperature=1.0', 'temperature=0.1'])

    plt.figure(3); plt.clf()
    x_gen_final_np = x_gen_final[:,0,:].squeeze().detach().numpy()
    X1 = 20 * np.log10(np.abs(np.fft.rfft(x_gen_final_np)))
    fbins = X1.shape[0]
    f = np.linspace(0, fs/2, fbins)
    plt.plot(f, X1)
    x_gen_cold_np = x_gen_cold[:,0,:].squeeze().detach().numpy()
    X2 = 20 * np.log10(np.abs(np.fft.rfft(x_gen_cold_np)))
    plt.plot(f, X2)
    plt.title("generated signal")
    plt.legend(['temperature=1.0', 'temperature=0.1'])

    plt.show()
