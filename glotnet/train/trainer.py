import torch

from .config import TrainerConfig
from glotnet.model.feedforward.wavenet import WaveNet
from glotnet.losses.distributions import Distribution, GaussianDensity


class Trainer():

    def __init__(self, config: TrainerConfig,
                 data_loader: torch.utils.data.DataLoader):

        self.config = config
        self.criterion: Distribution = self.create_criterion()
        self.model: WaveNet = self.create_model()
        self.optim: torch.optim.Optimizer = self.create_optimizer()

        self.data_loader = data_loader
        self.iter_global = 0
        self.iter = 0

    def create_model(self) -> WaveNet:
        """ Create model instance from config """
        cfg = self.config
        model = WaveNet(input_channels=cfg.input_channels,
                        output_channels=self.criterion.params_dim,
                        residual_channels=cfg.residual_channels,
                        skip_channels=cfg.skip_channels,
                        kernel_size=cfg.filter_width,
                        dilations=cfg.dilations,
                        causal=True,
                        activation=cfg.activation,
                        use_residual=cfg.use_residual,
                        cond_channels=cfg.cond_channels)
        return model

    def create_criterion(self) -> Distribution:
        """ Create scoring distribution instance from config """
        distribution = Trainer.distributions.get(
            self.config.distribution, None)
        if distribution is None:
            raise NotImplementedError(
                f"Distribution {self.config.distribution} not supported")
        dist = distribution()
        return dist

    def create_optimizer(self) -> torch.optim.Optimizer:
        """ Create optimizer instance from config """
        cfg = self.config
        Optimizer = Trainer.optimizers.get(self.config.optimizer, None)
        if Optimizer is None:
            raise NotImplementedError(
                f"Optimizer '{self.config.optimizer}' not supported")
        optim = Optimizer(self.model.parameters(), lr=cfg.learning_rate)
        return optim

    def resume(self, model_state_dict, optim_state_dict=None, iter=0):
        """ Resume training

        """
        self.model.load_state_dict(model_state_dict)
        if optim_state_dict is not None:
            self.optim.load_state_dict(optim_state_dict)
        self.iter_global = iter

    def fit(self, num_iters: int = 1, global_iter_max=None):
        while self.iter < num_iters:
            for minibatch in self.data_loader:
                x = minibatch[0]
                x_curr = x[:, :, 1:]
                x_prev = x[:, :, :-1]

                params = self.model(x_prev)

                nll = self.criterion.nll(x=x_curr, params=params)

                loss = nll.mean()
                loss.backward()

                self.optim.step()
                self.optim.zero_grad()

                self.iter += 1
                self.iter_global += 1
                if self.iter >= num_iters:
                    break
                if (global_iter_max is not None
                        and self.iter_global >= global_iter_max):
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
