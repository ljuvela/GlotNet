import torch

from torch.utils.tensorboard import SummaryWriter

from glotnet.config import Config

from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.data.audio_dataset import AudioDataset

from glotnet.model.autoregressive.glotnet import GlotNetAR
from glotnet.sigproc.lpc import LinearPredictor
from glotnet.sigproc.emphasis import Emphasis


from typing import Union

DeviceType = Union[str, torch.device]

class TrainerGlotNet(TrainerWaveNet):

    optim : torch.optim.Optimizer

    def __init__(self,
                 config: Config,
                 device: DeviceType = 'cpu'):
        """ Init GlotNet Trainer """
        super().__init__(config=config, device=device)
        
        if config.input_channels % 3 != 0:
            raise ValueError("Input channels must be divisible by 3")
        self.sample_after_filtering = config.glotnet_sample_after_filtering
        self.lpc = LinearPredictor(n_fft=config.n_fft,
            hop_length=config.hop_length,
            win_length=config.win_length,
            order=config.lpc_order).to(device)
    
    @property
    def model_ar(self):
        cfg = self.config
        cond_net = self.model.cond_net
        wavenet_cond_channels = cfg.cond_channels if cond_net is None else cond_net.output_channels
        distribution = self.criterion
        # TODO: teacher forcing and AR inference should be in the same model!
        if not hasattr(self, '_model_ar'):
            self.model_ar = GlotNetAR(
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
                lpc_order=cfg.lpc_order,
                hop_length=cfg.hop_length,
                cond_net=cond_net,
                sample_after_filtering=self.sample_after_filtering)
        return self._model_ar

    def generate(self,
                 dataset: AudioDataset,
                 temperature_voiced: torch.Tensor = None,
                 temperature_unvoiced: torch.Tensor = None
                 ):
        """ Generate samples in autoregressive inference mode
        
        Args:
            dataset: dataset to generate from (defaut: validation dataset)

            temperature_voiced: scaling factor for sampling noise in voiced regions
        """

        minibatch = dataset.__getitem__(0)
        x, c = self._unpack_minibatch(minibatch)
        c = c.unsqueeze(0)
        if self.model.cond_net is not None:
            c = self.model.cond_net(c)
        x = x.unsqueeze(0)
        x = x.to('cpu')
        if c is not None:
            c = torch.nn.functional.interpolate(
                        input=c, size=x.size(-1), mode='linear')
            c = c.to('cpu')

        temperature = self._temperature_from_voicing(c, temperature_voiced, temperature_unvoiced)

        model_ar = self.model_ar
        model_ar.load_state_dict(self.model.state_dict(), strict=False)
        x_emph = self.pre_emphasis.emphasis(x)
        a = self.lpc.estimate(x_emph[:, 0, :])

        output = model_ar.inference(
            input=torch.zeros_like(x),
            a=a, cond_input=c,
            temperature=temperature)
        output = output[:, :, model_ar.receptive_field:] # remove padding

        norm = output.abs().max()
        if norm > 1.0:
            print(f"output abs max was {norm}")
            output = output / norm
        return output.clamp(min=-0.99, max=0.99)


    def fit(self, num_iters: int = 1, global_iter_max=None):
        self.iter = 0
        stop = False
        while not stop:
            for minibatch in self.data_loader_training:
                x, c = self._unpack_minibatch(minibatch)
                x_emph = self.pre_emphasis.emphasis(x)

                # estimate lpc coefficients
                a = self.lpc.estimate(x_emph[:, 0, :])
                
                # clean excitation for target
                e_clean = self.lpc.inverse_filter(x, a)

                # add noise to signal
                x_noisy = x + (4.0 / 2 ** 16) * torch.randn_like(x)
                x_clean = x

                # get prediction signal
                p = self.lpc.prediction(x, a)

                # noisy error signal (residual)
                e_noisy = x_noisy - p

                e_curr = e_clean[:, :, 1:]
                e_prev = e_noisy[:, :, :-1]
                x_curr = x_clean[:, :, 1:]
                x_prev = x_noisy[:, :, :-1]
                p_curr = p[:, :, 1:]

                input = torch.cat([e_prev, p_curr, x_prev], dim=1)

                # TODO: move to model
                if self.model.cond_net is not None:
                    c = self.model.cond_net(c)

                if c is not None:
                    c = torch.nn.functional.interpolate(
                        input=c, size= x.size(-1), mode='linear')
                    # trim last sample to match x_prev size
                    c = c[..., :-1]

                params = self.model(input, c)

                if self.sample_after_filtering:
                    params = params + p_curr
                    loss = self.criterion(x=x_curr[..., self.config.padding:],
                                        params=params[..., self.config.padding:])
                else:
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

                if self.iter % self.config.validation_interval == 0:
                    self.writer.add_audio("excitation_clean", e_clean[0, 0, :], self.iter_global, sample_rate=self.config.sample_rate)
                    self.writer.add_audio("excitation_noisy", e_noisy[0, 0, :], self.iter_global, sample_rate=self.config.sample_rate)

                self.iter += 1
                self.iter_global += 1
                if self.iter >= num_iters:
                    stop = True
                    break
                if (global_iter_max is not None
                        and self.iter_global >= global_iter_max):
                    stop = True
                    break

