CONDA_BASE="$( conda info --base 2>/dev/null || echo "$HOME/anaconda3")"
CONDA_SCRIPT="$CONDA_BASE/etc/profile.d/conda.sh"
source "$CONDA_SCRIPT"
conda activate "$1"
