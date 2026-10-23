#!/bin/bash
#SBATCH --account=chenyijiang
#SBATCH --partition=hpg-default
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=150gb
#SBATCH --time=12:00:00
#SBATCH --job-name=sp_reg
#SBATCH --output=slurm_logs/sp_reg-%j.out
#
# Run pipeline step 3 (registration) for ONE TMA as a SLURM job.
#   sbatch -J sp_reg-<tma> -o slurm_logs/sp_reg-<tma>-%j.out \
#       shell/sbatch_registration.sh configs/<group>/<tma>.yaml
#
# --cpus-per-task matches registration.n_jobs in the configs (24): run_pipeline.py
# fans out to that many concurrent codex_registeration_batch.py subprocesses.
set -e
# Under sbatch the script runs from a copy in /var/spool/slurmd, so BASH_SOURCE does
# not point into the repo - use the submission directory, and fall back to the script
# location when run directly.
PROJECT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
[ -f "$PROJECT_DIR/run_pipeline.py" ] || { echo "run_pipeline.py not found under $PROJECT_DIR; submit from the project root" >&2; exit 1; }

# setup_conda.sh falls back to $HOME/anaconda3, which does not exist here - fail
# loudly instead of activating nothing.
command -v conda >/dev/null || { echo "conda not on PATH; submit from a shell with conda initialized" >&2; exit 1; }
source "$PROJECT_DIR/shell/setup_conda.sh" sp_process

# Each core is registered in its own subprocess, so keep the math libraries
# single-threaded to avoid oversubscribing the 24 allocated CPUs.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

cd "$PROJECT_DIR"
python run_pipeline.py --config "$1" --steps 3
