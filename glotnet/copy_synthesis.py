import argparse
import os
import torch
from glotnet.trainer.trainer import Trainer
from glotnet.config import Config

from glotnet.data.config import DataConfig
from glotnet.data.audio_dataset import AudioDataset

from glob import glob
import torchaudio

def parse_args():
    parser = argparse.ArgumentParser(
        description="GlotNet copy synthesis script")
    parser.add_argument('--config', required=True, type=str,
                        help='Configuration in json format')
    parser.add_argument('--model', required=True, type=str,
                        help="Model .pt file")
    parser.add_argument('--input_dir', required=True, type=str,
                        help="Audio file directory")
    parser.add_argument('--output_dir', required=True, type=str,
                        help="Output audio file directory")
    return parser.parse_args()

def main(args):
    
   
    config = Config.from_json(args.config)
    config.cond_channels = config.n_mels
    
    criterion = Trainer.create_criterion(config)
    model = Trainer.create_model(config, distribution=criterion)
    config.padding = model.receptive_field

    os.makedirs(args.output_dir, exist_ok=True)

    # audio_dir = '/Users/lauri/DATA/torchaudio/ARCTIC/cmu_us_slt_arctic/wav'
    input_dir = args.input_dir
    files = glob(os.path.join(input_dir, '*.wav'))
    files = files[:10]
    for f in files:
        config.segment_len = torchaudio.info(f).num_frames
        dataset = AudioDataset(config,
                               audio_dir=input_dir,
                               output_mel=True,
                               file_list=[f])
    

        trainer = Trainer(model=model,
                        criterion=criterion,
                        dataset=dataset,
                        config=config)

        # TODO: only load once
        trainer.load(model_path=args.model)

        x = trainer.generate(temperature=0.9)
        # trainer.writer.add_audio("generated audio_temp_1.0",
        #                          x[:, 0, :],
        #                          global_step=trainer.iter_global,
        #                          sample_rate=config.sample_rate)

        bname = os.path.basename(f)
        outfile = os.path.join(args.output_dir, bname)
        print(f"saving to {outfile}")
        torchaudio.save(outfile, x[0], sample_rate=config.sample_rate,
         bits_per_sample=16 ,encoding='PCM_S')

        # TODO: validation


if __name__ == "__main__":

    args = parse_args()
    main(args)