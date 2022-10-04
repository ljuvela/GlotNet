import argparse
import os
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
    trainer.config.to_json(os.path.join(trainer.writer.log_dir, 'config.json'))

    while trainer.iter_global < config.max_iters:
        ar_input = torch.zeros(1, 1, 2 * config.sample_rate)
        x = trainer.generate(ar_input, temperature=0.1)
        trainer.writer.add_audio("generated audio_temp_0.1",
                                 x[:, 0, :],
                                 global_step=trainer.iter_global,
                                 sample_rate=config.sample_rate)
        x = trainer.generate(ar_input, temperature=1.0)
        trainer.writer.add_audio("generated audio_temp_1.0",
                                 x[:, 0, :],
                                 global_step=trainer.iter_global,
                                 sample_rate=config.sample_rate)
        trainer.fit(num_iters=config.validation_interval,
                    global_iter_max=config.max_iters)
        torch.save(trainer.model.state_dict(), os.path.join(trainer.writer.log_dir, 'model-latest.pt'))
     
        # TODO: validation


if __name__ == "__main__":

    args = parse_args()
    main(args)