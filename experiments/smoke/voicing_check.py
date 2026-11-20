""" Science check for the Gaussian + Laplace mixture on the GlotNet excitation:
    is the Laplace component more responsible in voiced frames?

    usage: python -I voicing_check.py OUT [run]    (default run: gn_gl_shared)

    Per sample (teacher forced, training utterance):
      - prior weight of the Laplace component, w_L
      - posterior responsibility r_L = w_L p_L(e) / sum_k w_k p_k(e)
    averaged over 256-sample frames and grouped by
      - silence: frame RMS more than 35 dB below the loudest frame
      - voiced / unvoiced: the trainer's band-energy heuristic
        (Trainer._temperature_from_voicing: low-band mel energy > 1.1 x high band)
    The hypothesis predicts a higher r_L (and w_L) in voiced than in unvoiced
    frames. One overfitted utterance and a crude voicing decision: a smoke
    check, not evidence.

    Writes OUT/<run>/voicing_check.json, voicing_frames.csv and, if
    matplotlib is installed, voicing_check.png.
"""
import csv
import json
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))  # works with python -I
from runs import glotnet_teacher_forced, load_trainer  # noqa: E402

FRAME = 256
SILENCE_DB = -35.0


def main():
    out_dir = sys.argv[1]
    run = sys.argv[2] if len(sys.argv) > 2 else 'gn_gl_shared'
    run_dir = os.path.join(out_dir, run)

    with tempfile.TemporaryDirectory() as log_dir, \
            load_trainer(run_dir, out_dir, log_dir) as trainer:
        dist = trainer.criterion
        assert trainer.config.model_type == 'glotnet', "needs a GlotNet run"
        assert 'laplace' in dist.components, "needs a Laplace component"
        k_lap = dist.components.index('laplace')
        seg_len = trainer.config.segment_len

        # per-sample series on the global time axis of the utterance
        series = {}  # global index -> (x, r_L, w_L, voiced)
        with torch.no_grad():
            for i in range(len(trainer.dataset_training)):
                x, c = trainer.dataset_training[i]
                x, c = x.unsqueeze(0), c.unsqueeze(0)
                params, e_curr, x_curr, p_curr = glotnet_teacher_forced(trainer, x, c)
                if trainer.sample_after_filtering:
                    params, target = dist.shift_location(params, p_curr), x_curr
                else:
                    target = e_curr
                r = dist.responsibilities(target, params)[0, k_lap]
                logits = dist.split(params)[0]
                w = torch.softmax(logits, dim=1)[0, k_lap]
                c_up = torch.nn.functional.interpolate(c, size=x.size(-1), mode='linear')
                voiced = trainer._temperature_from_voicing(c_up, 1.0, 0.0)[0, 0]
                voiced = voiced[1:][trainer.config.padding:]  # align with x_curr
                # trimmed sample k of segment (start, stop) is at stop - seg_len + 1 + k
                _, _, stop = trainer.dataset_training.segment_index[i]
                first = stop - seg_len + 1
                for k in range(r.numel()):
                    t = first + k
                    if t >= 0:
                        series[t] = (x_curr[0, 0, k].item(), r[k].item(), w[k].item(),
                                     bool(voiced[k] > 0.5))

    t_sorted = sorted(series)
    xs = torch.tensor([series[t][0] for t in t_sorted])
    rs = torch.tensor([series[t][1] for t in t_sorted])
    ws = torch.tensor([series[t][2] for t in t_sorted])
    vs = torch.tensor([series[t][3] for t in t_sorted], dtype=torch.float32)
    n_frames = xs.numel() // FRAME
    frames = lambda v: v[:n_frames * FRAME].view(n_frames, FRAME)
    rms = frames(xs).pow(2).mean(dim=1).sqrt()
    level_db = 20 * torch.log10(rms / rms.max() + 1e-12)
    r_f, w_f = frames(rs).mean(dim=1), frames(ws).mean(dim=1)
    voiced_f = frames(vs).mean(dim=1) > 0.5
    group = ['silence' if level_db[f] < SILENCE_DB else
             ('voiced' if voiced_f[f] else 'unvoiced') for f in range(n_frames)]

    summary = {'run': run, 'frame_samples': FRAME, 'silence_db': SILENCE_DB,
               'components': list(dist.components), 'groups': {}}
    for g in ['voiced', 'unvoiced', 'silence']:
        idx = torch.tensor([f for f in range(n_frames) if group[f] == g], dtype=torch.long)
        summary['groups'][g] = {
            'frames': int(idx.numel()),
            'laplace_responsibility_mean': r_f[idx].mean().item() if idx.numel() else None,
            'laplace_weight_mean': w_f[idx].mean().item() if idx.numel() else None,
        }
    with open(os.path.join(run_dir, 'voicing_check.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(run_dir, 'voicing_frames.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time_s', 'level_db', 'group', 'laplace_responsibility',
                         'laplace_weight'])
        for fr in range(n_frames):
            writer.writerow([fr, round((t_sorted[0] + fr * FRAME) / trainer.config.sample_rate, 4),
                             round(level_db[fr].item(), 2), group[fr],
                             round(r_f[fr].item(), 4), round(w_f[fr].item(), 4)])
    print(json.dumps(summary, indent=2))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed: no plot (voicing_frames.csv has the data)")
        return
    t = [(t_sorted[0] + fr * FRAME) / trainer.config.sample_rate for fr in range(n_frames)]
    fig, ax = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
    ax[0].plot(t, level_db.tolist(), color='0.4')
    ax[0].set_ylabel('level (dB)')
    colors = {'voiced': 'tab:blue', 'unvoiced': 'tab:orange', 'silence': '0.7'}
    for g, col in colors.items():
        sel = [f for f in range(n_frames) if group[f] == g]
        ax[1].scatter([t[f] for f in sel], [r_f[f].item() for f in sel], s=8, color=col, label=g)
    ax[1].set_ylabel('Laplace responsibility')
    ax[1].set_xlabel('time (s)')
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(run_dir, 'voicing_check.png'), dpi=120)


if __name__ == '__main__':
    main()
