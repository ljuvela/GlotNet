import argparse
import os
import torch
from glotnet.trainer.trainer import Trainer as TrainerWaveNet
from glotnet.trainer.trainer_glotnet import Trainer as TrainerGlotNet
from glotnet.config import Config

from glotnet.data.config import DataConfig
from glotnet.data.audio_dataset import AudioDataset

from glotnet.sigproc.melspec import LogMelSpectrogram

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
    parser.add_argument('--max_files', default=None, type=int,
                        help="Maximum number of files to process")
    parser.add_argument('--temperature', default=1.0, type=float,
                        help="Temperature for sampling")
    return parser.parse_args()

def main(args):
    
    config = Config.from_json(args.config)

    os.makedirs(args.output_dir, exist_ok=True)

    melspec = LogMelSpectrogram(
        sample_rate=config.sample_rate,
        n_fft=config.n_fft,
        win_length=config.win_length,
        hop_length=config.hop_length,
        f_min=config.mel_fmin,
        f_max=config.mel_fmax,
        n_mels=config.n_mels,
    )

    # TODO: instantiate trainer here, load weights

    input_dir = args.input_dir
    files = glob(os.path.join(input_dir, '*.wav'))
    if args.max_files is not None:
        files = files[:args.max_files]

    #! Fix this later
    config.dataset_train_filelist = None

    dummy_dataset = AudioDataset(config,
                        audio_dir="",
                        file_list=[])

    if config.model_type == 'wavenet':
        trainer = TrainerWaveNet(config=config,
                        dataset=dummy_dataset,
                        device='cpu')
    elif config.model_type == 'glotnet':
        trainer = TrainerGlotNet(config=config,
                        dataset=dummy_dataset,
                        device='cpu')
    else:
        raise ValueError(f"Unknown model type {config.model_type}")
    
    # TODO: only load once
    trainer.load(model_path=args.model)

    for f in files:
        config.segment_len = torchaudio.info(f).num_frames
        dataset = AudioDataset(config,
                               audio_dir=input_dir,
                               transforms=melspec,
                               file_list=[f])
    
        trainer.set_dataset(dataset)

        x = trainer.generate(temperature=args.temperature)

        bname = os.path.basename(f)
        outfile = os.path.join(args.output_dir, bname)
        print(f"saving to {outfile}")
        torchaudio.save(outfile, x[0], sample_rate=config.sample_rate,
                        bits_per_sample=16 ,encoding='PCM_S')

        # TODO: validation


if __name__ == "__main__":

    args = parse_args()
    main(args)