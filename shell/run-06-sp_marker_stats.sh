set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/shell/setup_conda.sh" "/mnt/radonc-li02/private/lzx325/env/scimap2"  

data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"

tma_name="TA-29"
python "$PROJECT_DIR/src/sp_marker_stats.py" \
--input_dir "$data_dir/$tma_name/registration/aligned" \
--output_fp "$data_dir/$tma_name/processed/channel_stats.json" \
--tissue_mask_dir "$data_dir/$tma_name/tissue_detection/masks" \
--n_workers 8
