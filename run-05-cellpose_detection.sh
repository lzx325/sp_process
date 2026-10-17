set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/setup_conda.sh" "/mnt/radonc-li02/private/lzx325/env/cellpose"

data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"
python "$PROJECT_DIR/cellpose_detection.py" \
--input-dir "$data_dir/tma_cores/TA-29_registration/aligned/" \
--output-dir "$data_dir/tma_cores/cellpose_TA-29/masks" \
--dapi_channel 0 \
--n_gpu 1
