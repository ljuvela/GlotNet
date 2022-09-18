import torch
from torch.utils.data import TensorDataset, DataLoader

from glotnet.train.trainer import Trainer
from glotnet.train.config import TrainerConfig

def test_trainer():

    batch_size = 1
    timesteps = 100
    channels = 1

    # data
    f0 = 10
    fs = 100
    dur = int(1.0 * timesteps / fs) 
    t = torch.linspace(0, dur * fs, dur * fs) / fs
    x = torch.sin(2 * torch.pi * f0 * t)

    # data shape is ((batch_size, channels, timesteps)
    x = x.unsqueeze(0).unsqueeze(0)
    x_curr = x[:, :, 1:]
    x_prev = x[:, :, :-1]

    dataset = TensorDataset(x)
    config = TrainerConfig(batch_size=batch_size)
    criterion = Trainer.create_criterion(config)
    model = Trainer.create_model(config, criterion)

    trainer = Trainer(model=model,
                      criterion=criterion,
                      dataset=dataset,
                      config=config)

    likelihood_0 = trainer.log_prob(x_curr, x_prev)
    trainer.fit(num_iters=1)
    likelihood_1 = trainer.log_prob(x_curr, x_prev)

    assert likelihood_1 > likelihood_0, \
        "Training must improve likelihood"


if __name__ == "__main__":

    print("Testing unconditional training")
    test_trainer()
    print("-- OK!")