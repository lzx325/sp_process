set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source setup_conda.sh

data_dir="/media/lzx325/Elements/lzx325/tma2025_3/"

python "$PROJECT_DIR/sp_marker_stats.py" \
--input_dir "$data_dir/CODEX_registered" \
--output_fp "$data_dir/project/external_data/channel_stats.json" \
--tissue_mask_dir "$data_dir/project/external_data/tissue_detection/masks" \
--n_workers 8
