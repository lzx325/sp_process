set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source setup_conda.sh  "/mnt/radonc-li02/private/lzx325/env/scimap2"  

data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"

python "$PROJECT_DIR/sp_marker_stats.py" \
--input_dir "$data_dir/tma_cores/TA-29_registration/aligned/" \
--output_fp "$data_dir/tma_cores/processed_TA-29/channel_stats.json" \
--tissue_mask_dir "$data_dir/tma_cores/cellpose_TA-29/masks" \
--n_workers 8
