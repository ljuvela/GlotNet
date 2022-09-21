import argparse
from ast import parse

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
    parser.add_argument('--data_dir', help="Audio file directory for training")

def main(args):
    
    config = Config.from_json(args.config)
    criterion = Trainer.create_criterion(config)
    model = Trainer.create_model(config)
    config.padding = model.receptive_field

    audio_dir = '/Users/lauri/DATA/torchaudio/ARCTIC/cmu_us_slt_arctic/wav'
    dataset = AudioDataset(config, audio_dir=audio_dir)

    trainer = Trainer(model=model,
                      criterion=criterion,
                      dataset=dataset,
                      config=config)
    
    while trainer.iter_global < config.max_iters:
        trainer.fit(num_iters=config.logging_interval)


if __name__ == "__main__":

    args = parse_args()
    main(args)