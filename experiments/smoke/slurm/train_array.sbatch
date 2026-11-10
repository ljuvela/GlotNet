#!/bin/bash
# One smoke-test run per array task (task i = line i of `python runs.py`).
#
# Submit from the repository root:
#   UTTERANCE=/path/to/utt.wav OUT=/scratch/.../glotnet_smoke_mixture \
#     sbatch experiments/smoke/slurm/train_array.sbatch
# Optional: ITERS (default 5000), DEVICE (cpu; for cuda also request a GPU below),
# CONDA_ENV (default glotnet).
#
# Site-specific settings (account, partition, modules) are commented out:
# fill them in for your cluster.
#SBATCH --job-name=glotnet-smoke
#SBATCH --array=0-5
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=glotnet-smoke_%A_%a.out
##SBATCH --account=<project>
##SBATCH --partition=<partition>
##SBATCH --gpus=1          # only with DEVICE=cuda; the model is small, CPU is fine

set -euo pipefail
: "${UTTERANCE:?set UTTERANCE to the source wav}"
: "${OUT:?set OUT to the output directory}"
ITERS="${ITERS:-5000}"
DEVICE="${DEVICE:-cpu}"
cd "$SLURM_SUBMIT_DIR"   # repository root

# module load gcc cmake      # site specific, only if the env needs them at runtime
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-glotnet}"

mapfile -t RUNS < <(python -I experiments/smoke/runs.py)
if (( SLURM_ARRAY_TASK_ID >= ${#RUNS[@]} )); then
    echo "array index $SLURM_ARRAY_TASK_ID: only ${#RUNS[@]} runs"; exit 1
fi
RUN="${RUNS[$SLURM_ARRAY_TASK_ID]}"
echo "run $RUN on $(hostname), $(date), commit $(git rev-parse HEAD 2>/dev/null || cat .git/HEAD)"

mkdir -p "$OUT"
python -I experiments/smoke/train.py --run "$RUN" --utterance "$UTTERANCE" --out "$OUT" \
    --iters "$ITERS" --device "$DEVICE" --threads "${SLURM_CPUS_PER_TASK:-4}" \
    > "$OUT/train_$RUN.log" 2>&1
