set -e
PROJECT_DIR="/mnt/radonc-li02/private/lzx325/virtual-biomarking"
source "$PROJECT_DIR/shell_scripts/setup_conda_py.sh" "scimap2"

python "$PROJECT_DIR/tools/preprocess/he_codex/6_1_sp_other/qupath_tissue_detection_to_mask.py" \
--image_dir "/mnt/radonc-li02/public/he_codex/raw/tma2025/HE/" \
--geojson_dir "/mnt/radonc-li02/public/he_codex/raw/tma2025/project/external_data/tissue_detection/geojson" \
--output_dir "/mnt/radonc-li02/public/he_codex/raw/tma2025/project/external_data/tissue_detection/masks"