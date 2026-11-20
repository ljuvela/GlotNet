""" Smoke-test runs: overfit one utterance with each output distribution.

    `python runs.py` prints the run names, one per line (for job arrays).
    The settings below are the only place where the runs are defined.
"""
import os

import torch

from glotnet.config import Config
from glotnet.trainer.trainer import Trainer
from glotnet.trainer.trainer_glotnet import TrainerGlotNet

SAMPLE_RATE = 16000
UTTERANCE_FILE = 'utt16k.wav'

# shared by all runs (same as the categorical smoke test, glotnet_smoke)
BASE = dict(
    sample_rate=SAMPLE_RATE, segment_len=4000, batch_size=4,
    residual_channels=32, skip_channels=32, postnet_channels=64,
    dilations=(1, 2, 4, 8, 16, 32, 64, 128, 256),
    learning_rate=1e-3, pre_emphasis=0.85,
    n_mels=80, cond_channels=80, dataset_compute_mel=True,
    validation_interval=10**9)

RUNS = {
    # WaveNet on the pre-emphasised signal: compared in bits per mu-law sample
    'wn_gaussian': dict(distribution='gaussian'),
    'wn_gl': dict(distribution='mixture', mixture_components=('gaussian', 'laplace')),
    'wn_gl_shared': dict(distribution='mixture', mixture_components=('gaussian', 'laplace'),
                         mixture_shared_loc=True),
    'wn_mog4': dict(distribution='mixture', mixture_components=('gaussian',) * 4),
    'wn_dmol10': dict(distribution='mixture', mixture_components=('logistic',) * 10,
                      discretize_bits=16),
    # GlotNet (excitation) for the voicing check: Laplace responsibility vs voicing
    'gn_gl_shared': dict(model_type='glotnet', input_channels=3,
                         distribution='mixture', mixture_components=('gaussian', 'laplace'),
                         mixture_shared_loc=True),
}


def make_config(run: str, out_dir: str) -> Config:
    settings = dict(BASE)
    settings.update(RUNS[run])
    settings.update(
        dataset_audio_dir_training=os.path.join(out_dir, 'data'),
        dataset_filelist_training=[UTTERANCE_FILE],
        log_dir=os.path.join(out_dir, run, 'logs'))
    return Config(**settings)


def trainer_class(config: Config):
    return TrainerGlotNet if config.model_type == 'glotnet' else Trainer


def load_trainer(run_dir: str, out_dir: str, log_dir: str, device='cpu'):
    """ Trainer rebuilt from a run's checkpoint config, reading the utterance
        from out_dir/data (paths in the stored config may be from another machine) """
    model_pt = os.path.join(run_dir, 'model.pt')
    config = Trainer.config_from_checkpoint(model_pt)
    config.dataset_audio_dir_training = os.path.join(out_dir, 'data')
    config.dataset_filelist_training = [UTTERANCE_FILE]
    config.log_dir = log_dir
    trainer = trainer_class(config)(config=config, device=device)
    trainer.load(model_pt)
    trainer.model.eval()
    return trainer


def glotnet_teacher_forced(trainer, x, c):
    """ Teacher-forced GlotNet outputs for one batch, as in
        TrainerGlotNet.fit but with clean (noise-free) inputs.

    Returns (params, e_curr, x_curr, p_curr), receptive field trimmed.
    """
    x_emph = trainer.pre_emphasis.emphasis(x)
    a = trainer.lpc.estimate(x_emph[:, 0, :])
    e = trainer.lpc.inverse_filter(x, a)
    p = trainer.lpc.prediction(x, a)
    e_curr, e_prev = e[:, :, 1:], e[:, :, :-1]
    x_curr, x_prev = x[:, :, 1:], x[:, :, :-1]
    p_curr = p[:, :, 1:]
    model_input = torch.cat([e_prev, p_curr, x_prev], dim=1)
    if trainer.model.cond_net is not None:
        c = trainer.model.cond_net(c)
    c = torch.nn.functional.interpolate(c, size=x.size(-1), mode='linear')[..., :-1]
    params = trainer.model(model_input, c)
    pad = trainer.config.padding
    return params[..., pad:], e_curr[..., pad:], x_curr[..., pad:], p_curr[..., pad:]


if __name__ == '__main__':
    print('\n'.join(RUNS))
