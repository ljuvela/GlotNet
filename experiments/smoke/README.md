# Distribution smoke test

Overfit a small WaveNet / GlotNet on **one utterance** with each output
distribution, then check the path training → checkpoint → reload from the
stored config → C++ inference. Every run is scored in **bits per 8-bit mu-law
sample**, so continuous, discretized and categorical models are compared on the
same quantity. This is a smoke test, not a model-quality comparison: all
numbers are on the training utterance.

| file | what it does |
|---|---|
| `runs.py` | the run table (single source of truth); `python runs.py` lists the run names |
| `train.py` | trains one run and generates the utterance with C++ inference (`--prepare-only` writes `OUT/data/utt16k.wav`) |
| `evaluate.py` | teacher-forced bits per mu-law sample for all runs → `OUT/evaluation.json` and a markdown table |
| `voicing_check.py` | GlotNet Gaussian + Laplace: Laplace responsibility per frame vs voicing → `OUT/<run>/voicing_check.json`, `voicing_frames.csv` (+ `.png` if matplotlib is installed) |
| `run_local.sh` | everything on one machine (laptop, workstation, interactive node) |
| `slurm/train_array.sbatch` | one run per array task |
| `slurm/evaluate.sbatch` | evaluation + voicing check, after the array |

Runs (`runs.py`): `wn_gaussian`, `wn_gl` (Gaussian + Laplace), `wn_gl_shared`
(shared location), `wn_mog4` (4 Gaussians), `wn_dmol10` (10 discretized
logistics, 16 bit), `gn_gl_shared` (GlotNet, Gaussian + Laplace, shared
location). All share: residual/skip channels 32, dilations 1..256 (receptive field
1046), 80-band mel conditioning, pre-emphasis 0.85, Adam lr 1e-3, batch 4 × 4000
samples, 5000 iterations.

Outputs, per run, in `OUT/<run>/`: `model.pt` (self-describing checkpoint),
`history.json` (loss and distribution statistics every 50 iterations),
`run.json` (settings, timing, host, generation stats), `generated_T1.0.wav`,
`generated_T0.7v.wav` (temperature 0.7 in voiced, 1.0 in unvoiced regions),
`logs/` (TensorBoard).

## Data

LibriTTS-R dev-clean (CC BY 4.0), <https://www.openslr.org/141/>. One file of
1.36 GB; the smoke test needs a single utterance.

```bash
mkdir -p $DATA/LibriTTS_R && cd $DATA/LibriTTS_R
curl -sSL --fail -C - -o dev_clean.tar.gz https://www.openslr.org/resources/141/dev_clean.tar.gz
tar -xzf dev_clean.tar.gz
UTT=$DATA/LibriTTS_R/LibriTTS_R/dev-clean/1272/128104/1272_128104_000003_000002.wav
```

(On a cluster, download on a login/transfer node, but do not train there.)

## Local run

```bash
conda activate glotnet
experiments/smoke/run_local.sh $UTT $OUT 5000 3 3 cpu   # ITERS PARALLEL THREADS DEVICE
experiments/smoke/run_local.sh $UTT /tmp/smoke_tiny 20 3 3   # quick check first
```

Laptop (Apple M-series, 10 cores): 3 runs in parallel with 3 threads each, about
15 min per run at 5000 iterations.

## Moving to an HPC cluster (SLURM)

1. **Code.** Commit and push on the laptop, then clone on the cluster:
   ```bash
   git clone <repo-url> GlotNet && cd GlotNet
   git switch <branch>                       # the branch with this directory
   git submodule update --init --recursive   # third_party/eigen
   ```
2. **Environment** (conda; Python and build tools from conda-forge, torch from PyPI):
   ```bash
   module load gcc cmake        # site specific: a C++17 compiler is needed for the build
   conda env create -n glotnet -f environment.yml       # CUDA wheels
   # conda env create -n glotnet -f environment-cpu.yml # CPU-only nodes: smaller download
   conda activate glotnet
   pip install -v --no-build-isolation -e .   # builds the C++ extension against the installed torch
   ```
   Do not install conda packages that depend on `pytorch` (second copy of torch).
   If conda's `libstdc++` and the module's compiler disagree at import time,
   build and run with the same `gcc` module loaded.
3. **Tests**, in an interactive job, not on the login node:
   ```bash
   srun --cpus-per-task=4 --mem=8G --time=00:30:00 --pty bash
   conda activate glotnet
   pytest -q test                                # ~255 tests, ~10 s
   cmake -S . -B build-core -DGLOTNET_BUILD_PYTHON=OFF -DGLOTNET_BUILD_TESTS=ON
   cmake --build build-core -j4 && ctest --test-dir build-core   # torch-free C++ core
   ```
   On a GPU node, `test/test_trainer/test_trainer_device.py` also checks
   training on CUDA followed by C++ (CPU) generation.
4. **Data**: as above, into project/scratch storage.
5. **Run.** Edit the commented `#SBATCH --account/--partition` lines in both
   scripts for your site, then from the repository root:
   ```bash
   export UTTERANCE=$UTT OUT=/scratch/<project>/glotnet_smoke_mixture
   # quick check first (2 tasks, 20 iterations)
   ITERS=20 sbatch --array=0,5 --time=00:10:00 experiments/smoke/slurm/train_array.sbatch
   # full run
   jid=$(sbatch --parsable experiments/smoke/slurm/train_array.sbatch)
   sbatch --dependency=afterok:$jid experiments/smoke/slurm/evaluate.sbatch
   squeue -u $USER
   ```
   Logs: `glotnet-smoke_<jobid>_<task>.out` in the submit directory and
   `$OUT/train_<run>.log`. CPU is enough for these models (32 channels); for a
   GPU, set `DEVICE=cuda` and uncomment `--gpus=1`. Generation always runs on
   the CPU (C++ inference); `Trainer.generate` handles a model on the GPU.

## Notes for the next (Claude) session on the cluster

- **Ground rules from the user**:
  - Do not run git commands other than `git status`. Suggest the exact
    commands and let the user run them.
  - Install software only into conda envs, and ask before installing anything.
  - Test with a small input before scaling up.
  - Report job IDs, commands and output paths.
  - Never compute on the login node: use `srun` or `sbatch`.
- **State (2026-11-09)**, branch `categorical`:
  - Mu-law categorical output and the location-scale mixture are both done
    (TODO.md, "Mixture distributions" steps 0–5).
  - Step 6 is this smoke test. The laptop results are in
    `/Users/lauri/DATA/glotnet_smoke_mixture/RESULTS.md`, and summarised in
    TODO.md.
- **Where things are**:
  - Distributions:
    - `glotnet/losses/location_scale.py` (families)
    - `glotnet/losses/distributions.py` (Gaussian, Categorical, LocationScaleMixture)
    - C++: `include/glotnet/Distributions.h` and `ext/src/Distributions.cpp`
  - Config fields:
    - `distribution`, `num_bits`
    - `mixture_components`, `mixture_shared_loc`
    - `discretize_bits`, `entropy_floor_bits`
- **Likely next steps**:
  - A real multi-speaker run, which needs a training/validation file list and
    `glotnet/train.py`.
  - Listening tests.
  - Choosing what to ship as pretrained models (TODO.md, "Pretrained models":
    licences, where the weights live).
  - Batch-parallel C++ inference (TODO.md, "Inference").
