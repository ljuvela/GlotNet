import argparse

import torch
from glotnet.trainer.trainer import Trainer
from glotnet.config import Config

from glotnet.data.config import DataConfig
from glotnet.data.audio_dataset import AudioDataset

def parse_args():
    parser = argparse.ArgumentParser(
        description = "GlotNet main training script")
    parser.add_argument(
        '--config', help='configuration in json format')
    parser.add_argument('--log_dir')
    parser.add_argument('--saves_dir', help="Directory for saving model artefacts")
    parser.add_argument('--data_dir', type=str, help="Audio file directory for training")
    parser.add_argument('--device', type=str, default='cpu', help="Torch device string")
    return parser.parse_args()

def main(args):
    
    if args.config is None:
        config = Config()
    else:
        config = Config.from_json(args.config)

    
    criterion = Trainer.create_criterion(config)
    model = Trainer.create_model(config, distribution=criterion)
    config.padding = model.receptive_field

    # audio_dir = '/Users/lauri/DATA/torchaudio/ARCTIC/cmu_us_slt_arctic/wav'
    audio_dir = args.data_dir
    dataset = AudioDataset(config, audio_dir=audio_dir)

    device = torch.device(args.device)
    trainer = Trainer(model=model,
                      criterion=criterion,
                      dataset=dataset,
                      config=config,
                      device=device)
    
    while trainer.iter_global < config.max_iters:
        x = trainer.generate(torch.zeros(1, 1, 2 * config.sample_rate))
        trainer.writer.add_audio("generated audio",
                                 x[:, 0, :],
                                 global_step=trainer.iter_global,
                                 sample_rate=config.sample_rate)
        trainer.fit(num_iters=config.validation_interval,
                    global_iter_max=config.max_iters)
        # TODO: validation


if __name__ == "__main__":

    args = parse_args()
    main(args)