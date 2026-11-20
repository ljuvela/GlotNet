#!/bin/bash
# Evaluation and voicing check after the training array.
#
#   jid=$(UTTERANCE=... OUT=... sbatch --parsable experiments/smoke/slurm/train_array.sbatch)
#   OUT=... sbatch --dependency=afterok:$jid experiments/smoke/slurm/evaluate.sbatch
#SBATCH --job-name=glotnet-smoke-eval
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=glotnet-smoke-eval_%j.out
##SBATCH --account=<project>
##SBATCH --partition=<partition>

set -euo pipefail
: "${OUT:?set OUT to the output directory}"
cd "$SLURM_SUBMIT_DIR"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-glotnet}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

python -I experiments/smoke/evaluate.py "$OUT" | tee "$OUT/evaluation.log"
python -I experiments/smoke/voicing_check.py "$OUT" gn_gl_shared | tee "$OUT/voicing_check.log"
