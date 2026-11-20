#!/bin/bash
# Triton: one smoke-test run per array task (task i = line i of `python runs.py`).
# CPU only (32-channel models); partition is auto-selected by Slurm.
#
# Submit from the repository root. Quick check first (2 tasks, 20 iterations):
#   OUT=/scratch/elec/t412-speechsynth/ljuvela/glotnet_runs/smoke_tiny ITERS=20 \
#     sbatch --array=0,5 --time=00:15:00 -o "%x_%A_%a.out" \
#     experiments/smoke/slurm/triton_train_array.sbatch
# Full run (6 tasks, 5000 iterations; ~15 min each on a laptop with 3 threads):
#   jid=$(sbatch --parsable experiments/smoke/slurm/triton_train_array.sbatch)
#   sbatch --dependency=afterok:$jid experiments/smoke/slurm/triton_evaluate.sbatch
#
# Variables: OUT, UTTERANCE, ITERS (5000), SEED (0), SAMPLE_RATE (16000; LibriTTS-R is 24000),
# GLOTNET_ENV (glotnet-dev, see env.sh). Use a separate OUT per sample rate.
#SBATCH --job-name=glotnet-smoke
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=01:00:00
#SBATCH --output=glotnet-smoke_%A_%a.out

set -euo pipefail
OUT="${OUT:-/scratch/elec/t412-speechsynth/ljuvela/glotnet_runs/smoke_mixture}"
UTTERANCE="${UTTERANCE:-/scratch/elec/t412-speechsynth/DATA/LibriTTS-R/LibriTTS_R/dev-clean/1272/128104/1272_128104_000003_000002.wav}"
ITERS="${ITERS:-5000}"
SEED="${SEED:-0}"
SAMPLE_RATE="${SAMPLE_RATE:-16000}"
cd "$SLURM_SUBMIT_DIR"   # repository root
set +u; source env.sh; set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

mapfile -t RUNS < <(python -I experiments/smoke/runs.py)
if (( SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
    echo "array index $SLURM_ARRAY_TASK_ID: only ${#RUNS[@]} runs"; exit 1
fi
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "run $RUN on $(hostname), $(date), commit $(git rev-parse HEAD 2>/dev/null || cat .git/HEAD)"
echo "OUT=$OUT ITERS=$ITERS SEED=$SEED SAMPLE_RATE=$SAMPLE_RATE"

mkdir -p "$OUT"
python -I experiments/smoke/train.py --run "$RUN" --utterance "$UTTERANCE" --out "$OUT" \
    --iters "$ITERS" --seed "$SEED" --sample-rate "$SAMPLE_RATE" --device cpu --threads "$SLURM_CPUS_PER_TASK" \
    > "$OUT/train_$RUN.log" 2>&1
