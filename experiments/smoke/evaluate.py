""" Teacher-forced NLL of every smoke-test run in bits per mu-law sample

    usage: python -I evaluate.py OUT [run ...]   (default: all runs with a model.pt)

    Every distribution is scored on the same quantity: the probability mass
    it assigns to the 8-bit mu-law bin of each target sample (categorical:
    its own class probability; location-scale: CDF difference over the bin).
    WaveNet runs model the pre-emphasised signal and are comparable with
    each other. GlotNet runs are scored on the signal x (locations shifted by
    the LPC prediction), which is not pre-emphasised: not comparable with
    the WaveNet numbers.

    Also reports each run's own training criterion ('native' NLL, nats per
    sample: a density for continuous distributions, a bin mass when
    discretized). Writes OUT/evaluation.json and prints a markdown table.
"""
import json
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from runs import RUNS, glotnet_teacher_forced, load_trainer  # noqa: E402
from glotnet.losses.distributions import (CategoricalDensity, GaussianDensity,  # noqa: E402
                                          LocationScaleMixture)
from glotnet.sigproc import mulaw  # noqa: E402

MU = 255
LEVELS = mulaw.decode(torch.arange(MU + 1), MU).double()
EDGES = torch.cat([torch.tensor([-math.inf]), (LEVELS[1:] + LEVELS[:-1]) / 2,
                   torch.tensor([math.inf])]).double()


def mulaw_bin_log_mass(dist, target, params):
    """ log P(target's mu-law bin), shape (batch, 1, time), float64 """
    if isinstance(dist, CategoricalDensity):
        assert dist.num_bits == 8, "scoring assumes 8-bit mu-law"
        return -dist.nll(target, params).double()
    if isinstance(dist, GaussianDensity):
        # a one-component mixture with the same (m, log_s)
        dist, params = LocationScaleMixture(('gaussian',)), torch.cat(
            [torch.zeros_like(params[:, :1]), params], dim=1)
    q = mulaw.encode(target.double(), MU)
    return dist.log_mass(EDGES[q], EDGES[q + 1], params.double())


def evaluate_run(out_dir, run):
    run_dir = os.path.join(out_dir, run)
    with tempfile.TemporaryDirectory() as log_dir, \
            load_trainer(run_dir, out_dir, log_dir) as trainer:
        dist = trainer.criterion
        glotnet = trainer.config.model_type == 'glotnet'
        bits, native, n, beyond = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for i in range(len(trainer.dataset_training)):
                x, c = trainer.dataset_training[i]
                x, c = x.unsqueeze(0), c.unsqueeze(0)
                if glotnet:
                    params, e_curr, x_curr, p_curr = glotnet_teacher_forced(trainer, x, c)
                    if trainer.sample_after_filtering:
                        params = dist.shift_location(params, p_curr)
                        native_target = x_curr
                    else:
                        native_target = e_curr
                    native += dist.nll(native_target, params).sum().item()
                    if not trainer.sample_after_filtering:
                        params = dist.shift_location(params, p_curr)  # distribution of x
                    target = x_curr
                else:
                    x = trainer.pre_emphasis.emphasis(x)
                    c = torch.nn.functional.interpolate(c, size=x.size(-1), mode='linear')[..., :-1]
                    pad = trainer.config.padding
                    params = trainer.model(x[:, :, :-1], c)[..., pad:]
                    target = x[:, :, 1:][..., pad:]
                    native += dist.nll(target, params).sum().item()
                beyond += (target.abs() > 1.0).sum().item()
                bits -= mulaw_bin_log_mass(dist, target, params).sum().item() / math.log(2)
                n += target.numel()
    result = {'model_type': trainer.config.model_type,
              'distribution': type(dist).__name__,
              'settings': RUNS.get(run),
              'mulaw_bits_per_sample': bits / n,
              'native_nll_nats_per_sample': native / n,
              'samples': n,
              'samples_beyond_pm1': beyond}
    if isinstance(dist, LocationScaleMixture):
        result['components'] = list(dist.components)
    return result


def main():
    out_dir = sys.argv[1]
    runs = sys.argv[2:] or [r for r in RUNS
                            if os.path.exists(os.path.join(out_dir, r, 'model.pt'))]
    results = {}
    for run in runs:
        results[run] = evaluate_run(out_dir, run)
        print(run, json.dumps(results[run]), flush=True)
    with open(os.path.join(out_dir, 'evaluation.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print("\n| run | model | distribution | bits / mu-law sample | native NLL (nats) |")
    print("|---|---|---|---|---|")
    for run, r in results.items():
        print(f"| {run} | {r['model_type']} | {r['distribution']} "
              f"{tuple(r.get('components', ()))} | {r['mulaw_bits_per_sample']:.3f} "
              f"| {r['native_nll_nats_per_sample']:.3f} |")


if __name__ == '__main__':
    main()
