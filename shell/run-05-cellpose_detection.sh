set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/shell/setup_conda.sh" "/mnt/radonc-li02/private/lzx325/env/cellpose"

data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"
tma_name="TA-29"
python "$PROJECT_DIR/src/cellpose_detection.py" \
--input-dir "$data_dir/$tma_name/registration/aligned" \
--output-dir "$data_dir/$tma_name/cellpose/masks" \
--dapi_channel 0 \
--n_gpu 1
