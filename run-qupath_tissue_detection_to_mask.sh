set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source setup_conda.sh

data_dir="/media/lzx325/Elements/lzx325/tma2025_3"
python "$PROJECT_DIR/qupath_tissue_detection_to_mask.py" \
--image_dir "$data_dir/HE" \
--geojson_dir "$data_dir/project/external_data/tissue_detection/geojson" \
--output_dir "$data_dir/project/external_data/tissue_detection/masks"