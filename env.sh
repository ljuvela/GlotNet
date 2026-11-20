# Triton environment for GlotNet development and training.
# Usage (login node, interactive job, or Slurm script):
#     source env.sh
# Override the conda env name with GLOTNET_ENV=<name> if needed.

module load triton/2025.1-gcc gcc/13.3.0 mamba

source activate "${GLOTNET_ENV:-glotnet-dev}"

# C/C++ compiler for building the extension (pip install -e .)
export CC=gcc
export CXX=g++

# Keep pip's cache out of $HOME (torch CUDA wheels are several GB)
export PIP_CACHE_DIR="$WRKDIR/.cache/pip"

# Parallel C++ build: use the Slurm allocation if inside a job
export CMAKE_BUILD_PARALLEL_LEVEL="${SLURM_CPUS_PER_TASK:-4}"

# Note: torch 2.15 (CUDA 13.2) has no sm_70 kernels, so V100 GPUs do not
# work. Request A100 or newer in jobs, e.g. --gres=min-cuda-cc:80
