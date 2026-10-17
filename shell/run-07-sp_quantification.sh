set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/shell/setup_conda.sh" "/mnt/radonc-li02/private/lzx325/env/scimap2"  

if true; then
    data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"
    tma_name="TA-29"
    python "$PROJECT_DIR/src/sp_quantification.py" \
    --input_image_dir "$data_dir/$tma_name/registration/aligned" \
    --mask_dir "$data_dir/$tma_name/cellpose/masks" \
    --output_fp "$data_dir/$tma_name/processed/quantification.h5ad" \
    --n_workers 8 
fi

if false; then
    root_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_2/raw_scan/project/external_data/CODEX_TA-418_cores_matched"
    python "$PROJECT_DIR/src/sp_quantification.py" \
    --input_image_dir "$root_dir" \
    --mask_dir "$root_dir/cellpose" \
    --output_fp "$root_dir/clustering/quantification.h5ad" \
    --n_workers 8 

    root_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_TA-417_cores_matched"
    python "$PROJECT_DIR/src/sp_quantification.py" \
    --input_image_dir "$root_dir" \
    --mask_dir "$root_dir/cellpose" \
    --output_fp "$root_dir/clustering/quantification.h5ad" \
    --n_workers 8
fi

if false; then
    data_dir="/blue/chenyijiang/lzx325.stanford/data/tma2025_3/TA-29"
    python "$PROJECT_DIR/src/sp_quantification.py" \
    --input_image_dir "$data_dir/CODEX_registered" \
    --mask_dir "$data_dir/cellpose_masks" \
    --output_fp "$data_dir/clustering/quantification.h5ad" \
    --n_workers 8 \
    --slide-split
fi