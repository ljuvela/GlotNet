import tempfile
import torch
from torch.utils.data import TensorDataset, DataLoader

from glotnet.trainer.trainer import Trainer
from glotnet.config import Config

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

    dataset = TensorDataset(x)
    config = Config(batch_size=batch_size)
    criterion = Trainer.create_criterion(config)
    model = Trainer.create_model(config, criterion)

    trainer = Trainer(model=model,
                      criterion=criterion,
                      dataset=dataset,
                      config=config)

    x_curr = x[:, :, 1:]
    x_prev = x[:, :, :-1]
    likelihood_0 = trainer.log_prob(x_curr, x_prev)
    trainer.fit(num_iters=1)
    likelihood_1 = trainer.log_prob(x_curr, x_prev)

    assert likelihood_1 > likelihood_0, \
        "Training must improve likelihood"

def test_logging():

    batch_size = 4
    timesteps = 100
    channels = 1
    num_examples = 32

    x = torch.randn(num_examples, channels, timesteps)

    with tempfile.TemporaryDirectory() as dir:

        dataset = TensorDataset(x)
        config = Config(batch_size=batch_size)
        config.log_dir = dir
        criterion = Trainer.create_criterion(config)
        model = Trainer.create_model(config, criterion)


def test_resume_training():
    pass


if __name__ == "__main__":

    print("Testing unconditional training")
    test_trainer()
    print("-- OK!")