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
# mulaw.encode rounds in the companded domain: the boundary between classes
# k - 1 and k is the half-integer k - 1/2, expanded to the signal domain
_BOUNDARIES = mulaw.mu_law_decode(
    2.0 * (torch.arange(1, MU + 1, dtype=torch.float64) - 0.5) / MU - 1.0, MU)
EDGES = torch.cat([torch.tensor([-math.inf], dtype=torch.float64), _BOUNDARIES,
                   torch.tensor([math.inf], dtype=torch.float64)])


def discretized_bin_edges(lo, hi, bits, offset):
    """ Linear-cell edges covering the output levels inside the mu-law bin
        (lo, hi) of x. The levels of the discretized variable v are
        k * step - 1, k = 0..n, and x = v + offset (offset 0 when the grid is
        on x itself). Returns (v_lo, v_hi, empty); `empty` marks bins that
        contain no level and therefore have zero mass """
    n = 2 ** bits - 1
    step = 2.0 / n
    first = torch.ceil((lo - offset + 1.0) / step).clamp(min=0)
    last = (torch.ceil((hi - offset + 1.0) / step) - 1.0).clamp(max=n)
    inf = torch.full_like(first, math.inf)
    v_lo = torch.where(first <= 0, -inf, first * step - 1.0 - step / 2)
    v_hi = torch.where(last >= n, inf, last * step - 1.0 + step / 2)
    return v_lo, v_hi, first > last


def mulaw_bin_log_mass(dist, target, params, offset=0.0):
    """ log P(target's mu-law bin), shape (batch, 1, time), float64

        offset: x - v for a discretized dist whose grid is on v = x - offset
        (GlotNet excitation); params are then the parameters of v """
    if isinstance(dist, CategoricalDensity):
        assert dist.num_bits == 8, "scoring assumes 8-bit mu-law"
        return -dist.nll(target, params).double()
    if isinstance(dist, GaussianDensity):
        # a one-component mixture with the same (m, log_s)
        dist, params = LocationScaleMixture(('gaussian',)), torch.cat(
            [torch.zeros_like(params[:, :1]), params], dim=1)
    q = mulaw.encode(target.double(), MU)
    lo, hi = EDGES[q], EDGES[q + 1]
    params = params.double()
    if getattr(dist, 'discretize_bits', None) is not None:
        lo, hi, empty = discretized_bin_edges(lo, hi, dist.discretize_bits,
                                              torch.as_tensor(offset).double())
        log_mass = dist.log_mass(lo, hi, params)
        return torch.where(empty, torch.full_like(log_mass, -math.inf), log_mass)
    return dist.log_mass(lo, hi, params)


def evaluate_run(out_dir, run):
    run_dir = os.path.join(out_dir, run)
    with tempfile.TemporaryDirectory() as log_dir, \
            load_trainer(run_dir, out_dir, log_dir) as trainer:
        dist = trainer.criterion
        glotnet = trainer.config.model_type == 'glotnet'
        bits, native, n, beyond = 0.0, 0.0, 0, 0
        with torch.no_grad():
            for i in range(len(trainer.dataset_training)):
                offset = 0.0
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
                    target = x_curr
                    if not trainer.sample_after_filtering:
                        if getattr(dist, 'discretize_bits', None) is not None:
                            offset = p_curr  # grid stays on the excitation
                        else:
                            params = dist.shift_location(params, p_curr)  # distribution of x
                else:
                    x = trainer.pre_emphasis.emphasis(x)
                    c = torch.nn.functional.interpolate(c, size=x.size(-1), mode='linear')[..., :-1]
                    pad = trainer.config.padding
                    params = trainer.model(x[:, :, :-1], c)[..., pad:]
                    target = x[:, :, 1:][..., pad:]
                    native += dist.nll(target, params).sum().item()
                beyond += (target.abs() > 1.0).sum().item()
                bits -= mulaw_bin_log_mass(dist, target, params, offset).sum().item() / math.log(2)
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
