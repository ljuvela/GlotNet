#!/bin/bash
# Triton: evaluation (bits per mu-law sample) and voicing check after the training array.
#   sbatch --dependency=afterok:$jid experiments/smoke/slurm/triton_evaluate.sbatch
# Use the same OUT as for triton_train_array.sbatch.
#SBATCH --job-name=glotnet-smoke-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=glotnet-smoke-eval_%j.out

set -euo pipefail
OUT="${OUT:-/scratch/elec/t412-speechsynth/ljuvela/glotnet_runs/smoke_mixture}"
cd "$SLURM_SUBMIT_DIR"   # repository root
set +u; source env.sh; set -u
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

python -I experiments/smoke/evaluate.py "$OUT" | tee "$OUT/evaluation.log"
python -I experiments/smoke/voicing_check.py "$OUT" gn_gl_shared | tee "$OUT/voicing_check.log"
