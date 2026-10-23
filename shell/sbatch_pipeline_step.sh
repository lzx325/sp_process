#!/bin/bash
#SBATCH --account=chenyijiang
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --job-name=sp_step
#SBATCH --output=slurm_logs/sp_step-%j.out
#
# Run one pipeline step (or range) for ONE TMA as a SLURM job.
#   sbatch <resource flags> -J sp_s4-<tma> -o slurm_logs/sp_s4-<tma>-%j.out \
#       shell/sbatch_pipeline_step.sh configs/<group>/<tma>.yaml 4
#
# Resources differ per step, so they are passed on the sbatch command line rather
# than baked in here. Only run_pipeline.py itself runs in sp_process; each step's
# subprocess activates its own env (step 5 -> cellpose) via conda_envs in the config.
set -e
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$PROJECT_DIR/run_pipeline.py" ] || { echo "run_pipeline.py not found under $PROJECT_DIR; submit from the project root" >&2; exit 1; }

# setup_conda.sh falls back to $HOME/anaconda3, which does not exist here - fail
# loudly instead of activating nothing.
command -v conda >/dev/null || { echo "conda not on PATH; submit from a shell with conda initialized" >&2; exit 1; }
source "$PROJECT_DIR/shell/setup_conda.sh" sp_process

# Steps 6 and 7 fan out to n_workers processes, so keep the math libraries
# single-threaded to avoid oversubscribing the allocated CPUs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Step 5 allocates a few large tensors per core; let the caching allocator grow
# segments instead of fragmenting. Ignored by the non-GPU steps.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$PROJECT_DIR"
python run_pipeline.py --config "$1" --steps "$2"
