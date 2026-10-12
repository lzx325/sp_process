set -e
PROJECT_DIR="/mnt/radonc-li02/private/lzx325/virtual-biomarking"
source "$PROJECT_DIR/shell_scripts/setup_conda_py.sh" "scimap2"
if false; then
    python "$PROJECT_DIR/tools/preprocess/he_codex/6_1_sp_other/sp_quantification.py" \
    --input_image_dir "/mnt/radonc-li02/public/he_codex/raw/tma2025_3/CODEX_registered" \
    --mask_dir "/mnt/radonc-li02/public/he_codex/raw/tma2025_3/cellpose/masks" \
    --output_fp "/mnt/radonc-li02/public/he_codex/raw/tma2025_3/clustering/quantification.h5ad" \
    --n_workers 8 \
    --slide-split
fi

if true; then
    root_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_2/raw_scan/project/external_data/CODEX_TA-418_cores_matched"
    python "$PROJECT_DIR/tools/preprocess/he_codex/6_1_sp_other/sp_quantification.py" \
    --input_image_dir "$root_dir" \
    --mask_dir "$root_dir/cellpose" \
    --output_fp "$root_dir/clustering/quantification.h5ad" \
    --n_workers 8 

    root_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_TA-417_cores_matched"
    python "$PROJECT_DIR/tools/preprocess/he_codex/6_1_sp_other/sp_quantification.py" \
    --input_image_dir "$root_dir" \
    --mask_dir "$root_dir/cellpose" \
    --output_fp "$root_dir/clustering/quantification.h5ad" \
    --n_workers 8 
fi