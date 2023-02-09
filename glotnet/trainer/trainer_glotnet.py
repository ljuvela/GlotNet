import os
import torch

from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

from glotnet.config import Config
from glotnet.model.feedforward.wavenet import WaveNet
from glotnet.model.autoregressive.wavenet import WaveNetAR
from glotnet.losses.distributions import Distribution, GaussianDensity
from glotnet.data.audio_dataset import AudioDataset
from glotnet.sigproc.melspec import LogMelSpectrogram
from glotnet.sigproc.emphasis import Emphasis
from glotnet.sigproc.lpc import LinearPredictor

from typing import Union

DeviceType = Union[str, torch.device]

class Trainer(torch.nn.Module):

    optim : torch.optim.Optimizer

    def __init__(self,
                 config: Config,
                 dataset=None,
                 device: DeviceType = 'cpu'):
        """ Init GlotNet Trainer """
        super().__init__()
        self.device = device
        self.config = config
        self.config.input_channels = 3 * self.config.input_channels

        criterion= self._create_criterion()
        self.criterion = criterion.to(device)

        model = self._create_model(criterion)
        self.model = model.to(device)
        self.config.padding = model.receptive_field

        self.optim = self.create_optimizer()

        if dataset is None:
            if config.dataset_compute_mel:
                config.cond_channels = config.n_mels
                melspec = LogMelSpectrogram(
                    sample_rate=config.sample_rate,
                    n_fft=config.n_fft,
                    win_length=config.win_length,
                    hop_length=config.hop_length,
                    f_min=config.mel_fmin,
                    f_max=config.mel_fmax,
                    n_mels=config.n_mels)
            else:
                melspec = None

            self.dataset = AudioDataset(
                config=config,
                audio_dir=config.dataset_audio_dir,
                transforms=melspec)
        else:
            self.dataset = dataset

        self.pre_emphasis = Emphasis(alpha=config.pre_emphasis).to(device)
        self.lpc = LinearPredictor(n_fft=config.n_fft,
                                   hop_length=config.hop_length,
                                   win_length=config.win_length,
                                   order=10).to(device)

        self.data_loader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=config.shuffle,
            drop_last=True,
            num_workers=config.dataloader_workers)

        self.writer = self.create_writer()
        self.iter_global = 0
        self.iter = 0

    def to(self, device: DeviceType):
        self.device = device
        super().to(device)

    def _create_criterion(self) -> Distribution:
        """ Create scoring distribution instance from config """
        config = self.config
        distribution = Trainer.distributions.get(
            config.distribution, None)
        if distribution is None:
            raise NotImplementedError(
                f"Distribution {config.distribution} not supported")
        dist = distribution(entropy_floor=config.entropy_floor,
                            weight_entropy_penalty=config.loss_weight_entropy_hinge,
                            weight_nll=config.loss_weight_nll)
        return dist

    def _create_model(self, distribution: Distribution) -> WaveNet:
        """ Create model instance from config """
        cfg = self.config
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

    def generate(self, temperature: float = 1.0):
        """ Generate samples in autoregressive inference mode
        
        Args: 
            temperature: scaling factor for sampling noise
        """
        minibatch = self.dataset.__getitem__(0)
        x, c = self._unpack_minibatch(minibatch)
        c = c.unsqueeze(0)
        x = x.unsqueeze(0)
        x = x.to('cpu')
        if c is not None:
            c = torch.nn.functional.interpolate(
                        input=c, size= x.size(-1), mode='linear')
            c = c.to('cpu')

        cfg = self.config
        distribution = self.criterion
        # TODO: teacher forcing and AR inference should be in the same model!
        if not hasattr(self, 'model_ar'):
            self.model_ar = WaveNetAR(
                input_channels=cfg.input_channels,
                output_channels=distribution.params_dim,
                residual_channels=cfg.residual_channels,
                skip_channels=cfg.skip_channels,
                kernel_size=cfg.filter_width,
                dilations=cfg.dilations,
                causal=True,
                activation=cfg.activation,
                use_residual=cfg.use_residual,
                cond_channels=cfg.cond_channels)
            self.model_ar.pre_emphasis = Emphasis(alpha=cfg.pre_emphasis)
        model_ar = self.model_ar

        model_ar.load_state_dict(self.model.state_dict(), strict=False)
        model_ar.distribution.set_temperature(temperature)

        output = model_ar.forward(input=torch.zeros_like(x), cond_input=c)
        output = self.model_ar.pre_emphasis.deemphasis(output)
        return output.clamp(min=-0.99, max=0.99)

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
        writer = SummaryWriter(log_dir=self.config.log_dir)
        return writer

    def load(self, model_path: str, optim_path: str = None):
        """ Load trainer model and optimizer states """
        model_state_dict = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(model_state_dict)
        if optim_path is None:
            return
        optim_state_dict = torch.load(optim_path, map_location='cpu')
        self.optim.load_state_dict(optim_state_dict)

    def save(self, model_path: str, optim_path: str = None):
        """" Save model and optimizer state dictionaries"""
        torch.save(self.model.state_dict(), model_path)
        if optim_path is not None:
            torch.save(self.optim.state_dict(), optim_path)

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
                x = self.pre_emphasis.emphasis(x)

                # estimate lpc coefficients
                a = self.lpc.estimate(x[:, 0, :])
                
                # add noise to signal
                x = x + 1e-3 * torch.randn_like(x)

                # get prediction signal
                p = self.lpc.prediction(x, a)

                # error signal (residual)
                e = x - p

                e_curr = e[:, :, 1:]
                e_prev = e[:, :, :-1]

                x_prev = x[:, :, :-1]
                p_curr = p[:, :, 1:]

                input = torch.cat([e_prev, p_curr, x_prev], dim=1)

                if c is not None:
                    c = torch.nn.functional.interpolate(
                        input=c, size= x.size(-1), mode='linear')
                    # trim last sample to match x_prev size
                    c = c[..., :-1] 

                params = self.model(input, c)

                # discard non-valid samples (padding)
                loss = self.criterion(x=e_curr[..., self.config.padding:],
                                      params=params[..., self.config.padding:])
                loss.backward()

                self.batch_loss = loss

                # logging 
                self.writer.add_scalar("loss", loss.item(), global_step=self.iter_global)
                # TODO: log scale parameter
                self.writer.add_scalar(
                    "nll",
                    self.criterion.batch_nll.mean().item(),
                    global_step=self.iter_global)

                penalty = self.criterion.batch_penalty.mean().item()
                self.writer.add_scalar(
                    "entropy_floor_penalty",
                    penalty,
                    global_step=self.iter_global)

                self.writer.add_scalar(
                    "min_log_scale",
                    self.criterion.batch_log_scale.min().item(),
                    global_step=self.iter_global)

                self.optim.step()
                self.optim.zero_grad()

                if self.iter % 100 == 0:
                    print(f"Iter {self.iter_global}: loss = {loss.item()}")
                    print(f"     penalty = {penalty}, min log scale = {self.criterion.batch_log_scale.min().item()}")

                self.iter += 1
                self.iter_global += 1
                if self.iter >= num_iters:
                    stop = True
                    break
                if (global_iter_max is not None
                        and self.iter_global >= global_iter_max):
                    stop = True
                    break



    optimizers = {
        "adam": torch.optim.Adam
    }

    distributions = {
        "gaussian": GaussianDensity
    }

    schedulers = {
        "cyclic" : torch.optim.lr_scheduler.CyclicLR,
    }
