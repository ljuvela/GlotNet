import torch
from glotnet.model.feedforward.wavenet import WaveNet
from glotnet.model.autoregressive.wavenet import WaveNetAR
from glotnet.losses.distributions import GaussianDensity

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

    # model
    model = WaveNet(input_channels=1, output_channels=2, residual_channels=4,
                    skip_channels=4, kernel_size=3, dilations=[1, 2, 4, 8, 16])

    model_ar = WaveNetAR(input_channels=1, output_channels=2, residual_channels=4,
                    skip_channels=4, kernel_size=3, dilations=[1, 2, 4, 8, 16])

    # method 1
    # for p_src, p_dst in zip(model.parameters(), model_ar.parameters()):
    #     p_dst.data = p_src.data
    # method 2 
    model_ar.load_state_dict(model.state_dict())
    x_gen_init = model_ar.forward(timesteps=5000)


    # optimizer
    optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    # loss
    criterion = GaussianDensity()

    # iterate
    max_iters = 2000
    for iter in range(max_iters):

        x_curr = x[:, :, 1:]
        x_prev = x[:, :, :-1]

        params = model(x_prev)

        nll = criterion.nll(x=x_curr, params=params)
        loss = nll.sum()
        loss.backward()

        m = nll

        if iter % 100 == 0:
            print(f"Loss {iter}: {loss}")

        optim.step()
        optim.zero_grad()

    print("Finished training")

    m = params[:, 0:1, :]
    log_sig = params[:, 1:2, :]

    model_ar.load_state_dict(model.state_dict())
    x_gen_final = model_ar.forward(timesteps=5000)

    from matplotlib import pyplot as plt

    plt.plot(x_curr.squeeze())
    plt.plot(m.squeeze().detach())
    plt.legend(['ref', 'model mean'])
    plt.show()







    # train()