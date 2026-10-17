set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source setup_conda.sh "/mnt/radonc-li02/private/lzx325/env/scimap2"  

sample_name="TA-29"
data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"

python -u "$PROJECT_DIR/sp_cell_type_clustering.py" \
--input_anndata_fp "$data_dir/tma_cores/processed_TA-29/quantification.h5ad" \
--output_dir "$data_dir/tma_cores/processed_TA-29/" \
--run_preprocess \
--run_leiden \
--run_umap 
