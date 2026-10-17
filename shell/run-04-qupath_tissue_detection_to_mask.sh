set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/shell/setup_conda.sh" "/mnt/radonc-li02/private/lzx325/env/scimap2"  

data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"
tma_name="TA-29"
python "$PROJECT_DIR/src/qupath_tissue_detection_to_mask.py" \
--image_dir "$data_dir/$tma_name/HE_cores_matched" \
--geojson_dir "$data_dir/$tma_name/project/external_data/tissue_detection/geojson" \
--output_dir "$data_dir/$tma_name/tissue_detection/masks"