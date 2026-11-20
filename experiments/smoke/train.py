""" Train one smoke-test run, save the checkpoint, reload it from its stored
    config and generate the whole utterance with C++ inference.

    usage:
      python -I train.py --prepare-only --utterance U.wav --out OUT
      python -I train.py --run wn_gl --utterance U.wav --out OUT [--iters 5000]
                         [--device cpu|cuda|mps] [--threads 4] [--seed 0]

    Writes OUT/data/utt16k.wav (once) and OUT/<run>/{model.pt, history.json,
    run.json, generated_T1.0.wav, generated_T0.7v.wav, logs/}.
"""
import argparse
import json
import math
import os
import platform
import sys
import time

import torch
import torchaudio.functional as AF

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from runs import RUNS, SAMPLE_RATE, UTTERANCE_FILE, make_config, trainer_class  # noqa: E402
from glotnet.data import audio_io  # noqa: E402
from glotnet.trainer.trainer import Trainer  # noqa: E402


def prepare_utterance(src: str, out_dir: str) -> str:
    """ Resample to 16 kHz and clip to +-0.99; atomic, so parallel jobs can call it """
    data_dir = os.path.join(out_dir, 'data')
    path = os.path.join(data_dir, UTTERANCE_FILE)
    if os.path.exists(path):
        return path
    os.makedirs(data_dir, exist_ok=True)
    x, fs_orig = audio_io.load(src)
    x = AF.resample(x, fs_orig, SAMPLE_RATE).clamp(-0.99, 0.99)
    tmp = f"{path}.{os.getpid()}.tmp.wav"
    audio_io.save(tmp, x, sample_rate=SAMPLE_RATE)
    os.replace(tmp, path)
    print(f"utterance {src}: {x.size(-1) / SAMPLE_RATE:.2f} s at {SAMPLE_RATE} Hz, "
          f"peak {x.abs().max():.3f} -> {path}", flush=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', choices=sorted(RUNS))
    parser.add_argument('--utterance', required=True, help='source wav (any sample rate)')
    parser.add_argument('--out', required=True, help='output directory, shared by all runs')
    parser.add_argument('--iters', type=int, default=5000)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()

    utt_path = prepare_utterance(args.utterance, args.out)
    if args.prepare_only:
        return
    if args.run is None:
        parser.error('--run is required unless --prepare-only')

    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    config = make_config(args.run, args.out)
    run_dir = os.path.join(args.out, args.run)
    os.makedirs(run_dir, exist_ok=True)

    t0 = time.time()
    with trainer_class(config)(config=config, device=args.device) as trainer:
        dist = trainer.criterion
        print(f"{args.run}: {type(dist).__name__}, params_dim {dist.params_dim}, "
              f"receptive field {trainer.model.receptive_field}, "
              f"{len(trainer.dataset_training)} segments, device {args.device}", flush=True)
        history = []
        chunk = 50
        while trainer.iter_global < args.iters:
            trainer.fit(num_iters=min(chunk, args.iters - trainer.iter_global))
            nll = trainer.criterion.batch_nll.mean().item()
            entry = {'iter': trainer.iter_global, 'loss': trainer.batch_loss.item(),
                     'nll_nats': nll, 'time_s': round(time.time() - t0, 1)}
            entry.update(trainer.criterion.log_stats())
            history.append(entry)
            print(f"iter {trainer.iter_global}: loss {entry['loss']:.4f}, nll {nll:.4f} nats, "
                  f"{entry['time_s']:.0f} s", flush=True)
            if not math.isfinite(entry['loss']):
                raise RuntimeError(f"non-finite loss at iter {trainer.iter_global}")
        model_pt = os.path.join(run_dir, 'model.pt')
        trainer.save(model_pt)
    train_time = time.time() - t0
    with open(os.path.join(run_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)

    # reload from the checkpoint's own config, generate the whole utterance (CPU)
    saved = Trainer.config_from_checkpoint(model_pt)
    assert saved.distribution == config.distribution
    x = audio_io.load(utt_path)[0]
    with trainer_class(saved)(config=saved, device='cpu') as trainer2:
        trainer2.load(model_pt)
        padding = trainer2.config.padding  # receptive field
        trainer2.config.segment_len = ((x.size(-1) - padding)
                                       // saved.hop_length * saved.hop_length)
        gen_data = trainer2.create_dataset_training()
        generated = {}
        for name, t_voiced, t_unvoiced in [('T1.0', 1.0, 1.0), ('T0.7v', 0.7, 1.0)]:
            torch.manual_seed(1)
            t1 = time.time()
            y = trainer2.generate(dataset=gen_data, temperature_voiced=t_voiced,
                                  temperature_unvoiced=t_unvoiced)
            path = os.path.join(run_dir, f'generated_{name}.wav')
            audio_io.save(path, y[0], sample_rate=SAMPLE_RATE)
            generated[name] = {'seconds_audio': y.size(-1) / SAMPLE_RATE,
                               'seconds_generation': round(time.time() - t1, 2),
                               'finite': bool(torch.isfinite(y).all()),
                               'rms': y.pow(2).mean().sqrt().item(),
                               'peak': y.abs().max().item()}
            print(f"generated {path}: {generated[name]}", flush=True)

    with open(os.path.join(run_dir, 'run.json'), 'w') as f:
        json.dump({'run': args.run, 'settings': RUNS[args.run], 'iters': args.iters,
                   'seed': args.seed, 'device': args.device, 'threads': args.threads,
                   'train_time_s': round(train_time, 1), 'host': platform.node(),
                   'torch': torch.__version__, 'generated': generated},
                  f, indent=2)
    print("done", flush=True)


if __name__ == '__main__':
    main()
