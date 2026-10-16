set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source setup_conda.sh

data_dir="/blue/chenyijiang/lzx325.stanford/data/sword/downstream/lilab_tma2025_ovary"

python "$PROJECT_DIR/sp_marker_stats.py" \
--input_dir "$data_dir/raw/codex" \
--output_fp "$data_dir/processed/channel_stats.json" \
--tissue_mask_dir "$data_dir/processed/segmentation/masks" \
--n_workers 8
