import os
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from glotnet.config import Config
from glotnet.model.feedforward.wavenet import WaveNet
from glotnet.model.autoregressive.wavenet import WaveNetAR
from glotnet.losses.distributions import Distribution, GaussianDensity

class Trainer(torch.nn.Module):

    def __init__(self,
                 model: WaveNet,
                 criterion: Distribution,
                 dataset: Dataset,
                 config: Config,
                 device: torch.device = torch.device('cpu')):
        """ Init GlotNet Trainer """
        super().__init__()
        self.config = config
        self.criterion = criterion
        self.model = model
        self.optim = self.create_optimizer()
        self.dataset = dataset
        self.data_loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            drop_last=True)
        self.writer = self.create_writer()
        self.iter_global = 0
        self.iter = 0
        self.device = device

    def to(self, device: torch.device):
        self.device = device
        super().to(device)

    def create_criterion(config: Config) -> Distribution:
        """ Create scoring distribution instance from config """
        distribution = Trainer.distributions.get(
            config.distribution, None)
        if distribution is None:
            raise NotImplementedError(
                f"Distribution {config.distribution} not supported")
        dist = distribution()
        return dist

    def create_model(config: Config, distribution: Distribution) -> WaveNet:
        """ Create model instance from config """
        cfg = config
        # TODO: distribution should be a part of the model
        model = WaveNet(input_channels=cfg.input_channels,
                        output_channels=distribution.params_dim,
                        residual_channels=cfg.residual_channels,
                        skip_channels=cfg.skip_channels,
                        kernel_size=cfg.filter_width,
                        dilations=cfg.dilations,
                        causal=True,
                        activation=cfg.activation,
                        use_residual=cfg.use_residual,
                        cond_channels=cfg.cond_channels)
        return model

    def generate(self, input: torch.Tensor):
        """ Generate samples in autoregressive inference mode
        
        Args: input shape is (batch, channels, timesteps)
        """
        cfg = self.config
        distribution = self.criterion
        # TODO: teacher forcing and AR inference should be in the same model!
        model_ar = WaveNetAR(input_channels=cfg.input_channels,
                        output_channels=distribution.params_dim,
                        residual_channels=cfg.residual_channels,
                        skip_channels=cfg.skip_channels,
                        kernel_size=cfg.filter_width,
                        dilations=cfg.dilations,
                        causal=True,
                        activation=cfg.activation,
                        use_residual=cfg.use_residual,
                        cond_channels=cfg.cond_channels)

        model_ar.load_state_dict(self.model.state_dict(), strict=False)
        model_ar.distribution.set_temperature(0.1) # TODO: schedule?
        output = model_ar.forward(input=input)
        return output.clamp(min=0.99, max=0.99)

    def create_optimizer(self) -> torch.optim.Optimizer:
        """ Create optimizer instance from config """
        cfg = self.config
        Optimizer = Trainer.optimizers.get(self.config.optimizer, None)
        if Optimizer is None:
            raise NotImplementedError(
                f"Optimizer '{self.config.optimizer}' not supported")
        optim = Optimizer(self.model.parameters(), lr=cfg.learning_rate)
        return optim

    def create_scheduler(self) -> torch.optim.lr_scheduler._LRScheduler:
        """ Create learning rate sceduler """
        raise NotImplementedError("Learning rate scheduling not implemented yet")

    def create_writer(self) -> SummaryWriter:
        os.makedirs(self.config.log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=self.config.log_dir)
        return writer

    def resume(self, model_state_dict, optim_state_dict=None, iter=0):
        """ Resume training

        """
        self.model.load_state_dict(model_state_dict)
        if optim_state_dict is not None:
            self.optim.load_state_dict(optim_state_dict)
        self.iter_global = iter

    def _unpack_minibatch(self, minibatch):
        """ Unpack minibatch and move to appropriate device """
        if len(minibatch) == 1:
            x, = minibatch
            c = None
        elif len(minibatch) == 2:
            x, c = minibatch
            c = c.to(self.device)
        else:
            raise ValueError("")
        x = x.to(self.device)
        return x, c

    def fit(self, num_iters: int = 1, global_iter_max=None):
        self.iter = 0
        stop = False
        while not stop:
            for minibatch in self.data_loader:
                x, c = self._unpack_minibatch(minibatch)
                x_curr = x[:, :, 1:]
                x_prev = x[:, :, :-1]

                params = self.model(x_prev, c)

                # discard non-valid samples (padding)
                loss = self.criterion(x=x_curr[self.config.padding:],
                                      params=params[self.config.padding:])
                loss.backward()

                # logging 
                self.writer.add_scalar("loss", loss.item(), global_step=self.iter_global)
                # TODO: log scale parameter

                self.optim.step()
                self.optim.zero_grad()

                print(f"Iter {self.iter_global}: loss = {loss.item()}")

                self.iter += 1
                self.iter_global += 1
                if self.iter >= num_iters:
                    stop = True
                    break
                if (global_iter_max is not None
                        and self.iter_global >= global_iter_max):
                    stop = True
                    break


    def log_prob(self, x_curr: torch.Tensor,
                 x_prev: torch.Tensor,
                 cond: torch.Tensor = None) -> torch.Tensor:
        """ Log probablilty of observations """
        params = self.model.forward(x_prev, cond)
        nll = self.criterion.nll(x_curr, params)
        return nll.sum() * -1.0

    optimizers = {
        "adam": torch.optim.Adam
    }

    distributions = {
        "gaussian": GaussianDensity
    }

    schedulers = {
        "cyclic" : torch.optim.lr_scheduler.CyclicLR,
    }
