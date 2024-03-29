import argparse
import os
import torch
from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.trainer.trainer_glotnet import TrainerGlotNet
from glotnet.config import Config


from glotnet.data.config import DataConfig
from glotnet.data.audio_dataset import AudioDataset

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        description = "GlotNet main training script")
    parser.add_argument(
        '--config', help='configuration in json format')
    parser.add_argument('--log_dir')
    parser.add_argument('--saves_dir',
        help="Directory for saving model artefacts")
    parser.add_argument('--audio_dir_train', type=str, default=None,
        help="Audio file directory for training")
    parser.add_argument('--audio_dir_val', type=str, default=None,
        help="Audio file directory for validation")
    parser.add_argument('--mel_cond', type=bool, default=True, 
        help="Condition on Mel Spectrum")
    parser.add_argument('--device', type=str, default='cpu', 
        help="Torch device string")
    parser.add_argument('--model_pt', type=str, default=None, 
        help="Pre-trained model .pt file")
    parser.add_argument('--optim_pt', type=str, default=None, 
        help="Optimizer state dictionary .pt file (use to continue training)")
    parser.add_argument('--compile', type=bool, default=False,
        help="Compile model for accelerated training (requires torch >= 2.0.0)")
    return parser.parse_args(args=args)

def main(args):
    
    if args.config is None:
        config = Config()
    else:
        config = Config.from_json(args.config)

    if args.mel_cond:
        config.cond_channels = config.n_mels
        config.dataset_compute_mel = True
    else:
        config.cond_channels = None
        config.dataset_compute_mel = False

    if args.audio_dir_train is not None:
        config.dataset_audio_dir_training = args.audio_dir_train
    if args.audio_dir_val is not None:
        config.dataset_audio_dir_validation = args.audio_dir_val

    if config.model_type == 'glotnet':
        trainer = TrainerGlotNet(config=config, device=args.device)
    elif config.model_type == 'wavenet':
        trainer = TrainerWaveNet(config=config, device=args.device)
    else:
        raise NotImplementedError(f"Model type {config.model_type} not implemented")
    trainer.config.to_json(os.path.join(trainer.writer.log_dir, 'config.json'))

    if args.model_pt is not None:
        trainer.load(args.model_pt, args.optim_pt)

    if trainer.dataset_validation is not None:
        gen_data = trainer.dataset_validation
    else:
        gen_data = trainer.dataset_training

    if args.compile:
        trainer.model = torch.compile(trainer.model)

    loss_best = 1e9
    current_patience = config.max_patience

    while trainer.iter_global < config.max_iters:
        trainer.fit(num_iters=config.validation_interval,
                    global_iter_max=config.max_iters)
        trainer.save(
            model_path=os.path.join(trainer.writer.log_dir, 'model-latest.pt'),
            optim_path=os.path.join(trainer.writer.log_dir, 'optim-latest.pt'))
        x = trainer.generate(dataset=gen_data)
        trainer.writer.add_audio("generated audio_temp_1.0",
                                 x[:, 0, :],
                                 global_step=trainer.iter_global,
                                 sample_rate=config.sample_rate)

        if trainer.dataset_validation is None:
            print(f"Validation data not provided, skipping validation step")
        else:
            loss_valid = trainer.validate()
            if loss_valid < loss_best:
                current_patience = config.max_patience
                loss_best = loss_valid
                trainer.save(
                    model_path=os.path.join(trainer.writer.log_dir, 'model-best.pt'),
                    optim_path=os.path.join(trainer.writer.log_dir, 'optim-best.pt'))
            else:
                current_patience -= 1
                
            if current_patience <= 0:
                print(f"Early stopping at iteration {trainer.iter_global}")
                break


if __name__ == "__main__":

    args = parse_args()
    main(args)