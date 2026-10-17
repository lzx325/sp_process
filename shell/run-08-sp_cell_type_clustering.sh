set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/shell/setup_conda.sh" "/mnt/radonc-li02/private/lzx325/env/scimap2"  

data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"

tma_name="TA-29"
python -u "$PROJECT_DIR/src/sp_cell_type_clustering.py" \
--input_anndata_fp "$data_dir/$tma_name/processed/quantification.h5ad" \
--output_dir "$data_dir/$tma_name/processed/" \
--run_preprocess \
--run_leiden \
--run_umap 
