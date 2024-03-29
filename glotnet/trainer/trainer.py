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
import numpy as np

from typing import Union

DeviceType = Union[str, torch.device]
class Trainer(torch.nn.Module):

    optim : torch.optim.Optimizer

    def __init__(self,
                 config: Config,
                 device: DeviceType = 'cpu'):
        """ Init GlotNet Trainer """
        super().__init__()
        self.device = device
        self.config = config

        criterion= self._create_criterion()
        self.criterion = criterion.to(device)

        model = self._create_model(criterion)
        self.model = model.to(device)
        self.config.padding = model.receptive_field

        self.optim = self.create_optimizer()

        self.pre_emphasis = Emphasis(alpha=config.pre_emphasis).to(device)
        self.dataset_training : AudioDataset = None
        self.dataset_validation : AudioDataset = None
        self._data_loader_training : DataLoader = None
        self._data_loader_validation : DataLoader = None

        if config.dataset_filelist_training is not None:
            data_train = self.create_dataset_training()
            self.set_training_dataset(data_train)

        if config.dataset_filelist_validation is not None:
            data_val = self.create_dataset_validation()
            self.set_validation_dataset(data_val)

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
        if cfg.use_condnet:
            cond_net = WaveNet(
                input_channels=cfg.cond_channels,
                output_channels=cfg.residual_channels,
                residual_channels=cfg.condnet_residual_channels,
                skip_channels=cfg.condnet_skip_channels,
                kernel_size=cfg.condnet_filter_width,
                causal=cfg.condnet_causal,
                dilations=cfg.condnet_dilations)
        else:
            cond_net = None

        wavenet_cond_channels = cfg.cond_channels if cond_net is None else cond_net.output_channels

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
                        cond_channels=wavenet_cond_channels,
                        cond_net=cond_net)
        return model
    
    def _temperature_from_voicing(
            self, c, temperature_voiced:float=0.7, temperature_unvoiced:float=1.0):
        """ Simple voicing decision based on upper and lower band energies """

        if temperature_voiced is None:
            return None

        if self.dataset_training is not None and self.dataset_training.use_scaler:
            c_denorm = c * self.dataset_training.scaler_s + self.dataset_training.scaler_m
        else:
            c_denorm = c
        c_l, c_h = torch.chunk(c_denorm, dim=1, chunks=2)
        c_l = c_l.exp().sum(dim=1,keepdim=True)
        c_h = c_h.exp().sum(dim=1,keepdim=True)
        voiced = c_l > 1.1 * c_h 
        temperature = temperature_voiced * voiced + temperature_unvoiced * ~voiced
        return temperature

    @property
    def model_ar(self):
        cfg = self.config
        cond_net = self.model.cond_net
        wavenet_cond_channels = cfg.cond_channels if cond_net is None else cond_net.output_channels
        distribution = self.criterion
        # TODO: teacher forcing and AR inference should be in the same model!
        if not hasattr(self, '_model_ar'):
            self._model_ar = WaveNetAR(
                input_channels=cfg.input_channels,
                output_channels=distribution.params_dim,
                residual_channels=cfg.residual_channels,
                skip_channels=cfg.skip_channels,
                kernel_size=cfg.filter_width,
                dilations=cfg.dilations,
                causal=True,
                activation=cfg.activation,
                use_residual=cfg.use_residual,
                cond_channels=wavenet_cond_channels,
                cond_net=cond_net,)
        return self._model_ar

    def generate(self,
                 dataset: AudioDataset,
                 temperature_voiced: float = None,
                 temperature_unvoiced: float = None,
                 use_temperature_from_voicing: bool = True
                 ):
        """ Generate samples in autoregressive inference mode
        
        Args:
            dataset: dataset to generate from (defaut: validation dataset)
            temperature: scaling factor for sampling noise
        """

        minibatch = dataset.__getitem__(0)
        x, c = self._unpack_minibatch(minibatch)

        x = x.unsqueeze(0)
        x = x.to('cpu')
        if c is not None:
            c = c.unsqueeze(0)
            if self.model.cond_net is not None:
                c = self.model.cond_net(c)
            c = torch.nn.functional.interpolate(
                        input=c, size= x.size(-1), mode='linear')
            c = c.to('cpu')

        if use_temperature_from_voicing:
            temperature = self._temperature_from_voicing(c, temperature_voiced, temperature_unvoiced)
        else:
            temperature = temperature_voiced * torch.ones_like(x)

        model_ar = self.model_ar
        model_ar.load_state_dict(self.model.state_dict(), strict=False)

        output = model_ar.inference(
            input=torch.zeros_like(x),
            cond_input=c,
            temperature=temperature)
        output = output[:, :, model_ar.receptive_field:] # remove padding
        output = self.pre_emphasis.deemphasis(output)
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
            for minibatch in self.data_loader_training:
                x, c = self._unpack_minibatch(minibatch)
                x = self.pre_emphasis.emphasis(x)
                x_curr = x[:, :, 1:]
                x_prev = x[:, :, :-1]

                if self.model.cond_net is not None:
                    c = self.model.cond_net(c)

                if c is not None:
                    c = torch.nn.functional.interpolate(
                        input=c, size= x.size(-1), mode='linear')
                    # trim last sample to match x_prev size
                    c = c[..., :-1] 

                params = self.model(x_prev, c)

                # discard non-valid samples (padding)
                loss = self.criterion(x=x_curr[..., self.config.padding:],
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
        
    def validate(self):
        losses = []
        with torch.no_grad():
            for minibatch in self.data_loader_validation:
                x, c = self._unpack_minibatch(minibatch)
                x = self.pre_emphasis.emphasis(x)
                x_curr = x[:, :, 1:]
                x_prev = x[:, :, :-1]

                if self.model.cond_net is not None:
                    c = self.model.cond_net(c)

                if c is not None:
                    c = torch.nn.functional.interpolate(
                        input=c, size= x.size(-1), mode='linear')
                    # trim last sample to match x_prev size
                    c = c[..., :-1]

                params = self.model(x_prev, c)

                loss = self.criterion(x=x_curr[..., self.config.padding:],
                        params=params[..., self.config.padding:])
                losses.append(loss.item())

            mean_loss = np.mean(losses)
            self.writer.add_scalar("loss_validation", mean_loss, global_step=self.iter_global)

            return mean_loss


    def create_dataset_training(self) -> AudioDataset:
        config = self.config
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

        filelist = AudioDataset.read_filelist(
            audio_dir=config.dataset_audio_dir_training,
            filelist=config.dataset_filelist_training)
        dataset = AudioDataset(
            config=config,
            audio_dir=config.dataset_audio_dir_training,
            file_list=filelist,
            transforms=melspec)

        return dataset

    def create_dataset_validation(self) -> AudioDataset:
        config = self.config
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

        filelist_val = AudioDataset.read_filelist(
            audio_dir=config.dataset_audio_dir_validation,
            filelist=config.dataset_filelist_validation)
        dataset = AudioDataset(
            config=config,
            audio_dir=config.dataset_audio_dir_validation,
            file_list=filelist_val,
            transforms=melspec)
        return dataset


    @property
    def data_loader_training(self):
        if self._data_loader_training is None:
            raise ValueError("No training data loader defined, call set_training_dataset() first")
        return self._data_loader_training

    def set_training_dataset(self, dataset: AudioDataset = None):
        """ Set training dataset and create data loader """

        self.dataset_training = dataset
        if dataset is None:
            self._data_loader_training = None
        else:
            self._data_loader_training = DataLoader(
                self.dataset_training,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                drop_last=True,
                num_workers=self.config.dataloader_workers)

    @property
    def data_loader_validation(self):
        if self._data_loader_validation is None:
            raise ValueError("No training data loader defined, call set_training_dataset() first")
        return self._data_loader_validation

    def set_validation_dataset(self, dataset: AudioDataset = None):
        """ Set validation dataset and create data loader """

        self.dataset_validation = dataset
        if dataset is None:
            self._data_loader_validation = None
        else:
            self._data_loader_validation = DataLoader(
                self.dataset_training,
                batch_size=self.config.batch_size,
                shuffle=self.config.shuffle,
                drop_last=True,
                num_workers=self.config.dataloader_workers)


    optimizers = {
        "adam": torch.optim.Adam
    }

    distributions = {
        "gaussian": GaussianDensity
    }

    schedulers = {
        "cyclic" : torch.optim.lr_scheduler.CyclicLR,
    }

