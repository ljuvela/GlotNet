import soundfile as sf
import torch
import matplotlib.pyplot as plt

from glotnet.sigproc.lpc import LinearPredictor
from glotnet.sigproc.emphasis import Emphasis

if __name__ == '__main__':



    x_np, fs = sf.read('/Users/ljuvela/DATA/ARCTIC/cmu_us_slt_arctic/wav/arctic_a0001.wav')
    x_np = -x_np # polarity flip
    x = torch.from_numpy(x_np).unsqueeze(0).unsqueeze(0).float()

    # stft parameters
    n_fft = 512
    hop_length = 128
    win_length = 512
    order = 10

    emph = Emphasis(alpha=0.85)

    lpc = LinearPredictor(n_fft=n_fft, hop_length=hop_length, win_length=win_length, order=order)

    x_emph = emph.emphasis(x)

    a = lpc.estimate(x=x_emph[:, 0, :])
    p = lpc.prediction(x=x, a=a)
    e_inv = lpc.inverse_filter(x=x, a=a)

    e_diff = x - p

    plt.close('all')

    plt.plot(e_diff[0,0,:].detach().numpy())
    plt.plot(e_inv[0,0,:].detach().numpy())
    plt.plot(x[0,0,:].detach().numpy())
    plt.legend(['difference', 'inverse filtered', 'original'])
    plt.show()
