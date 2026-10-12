set -e
PROJECT_DIR="/mnt/radonc-li02/private/lzx325/virtual-biomarking"
source "$PROJECT_DIR/shell_scripts/setup_conda_py.sh" "scimap2"

python "$PROJECT_DIR/tools/preprocess/he_codex/6_1_sp_other/sp_marker_stats.py" \
--input_dir "/mnt/radonc-li02/public/he_codex/raw/tma2025/CODEX_registered" \
--output_fp "/mnt/radonc-li02/public/he_codex/raw/tma2025/project/external_data/channel_stats.json" \
--tissue_mask_dir "/mnt/radonc-li02/public/he_codex/raw/tma2025/project/external_data/tissue_detection/masks" \
--n_workers 8
