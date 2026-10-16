set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source setup_conda.sh

sample_name="TA-29"
data_dir="/blue/chenyijiang/lzx325.stanford/data/tma2025_3/TA-29"

python -u "$PROJECT_DIR/sp_cell_type_clustering.py" \
--input_anndata_fp "$data_dir/clustering/quantification_${sample_name}.h5ad" \
--output_dir "$data_dir/clustering" \
--run_preprocess \
--run_leiden \
--run_umap 
