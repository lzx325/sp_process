set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source setup_conda.sh

sample_name="$1"
python -u "$PROJECT_DIR/sp_cell_type_clustering.py" \
--input_anndata_fp "/mnt/radonc-li02/public/he_codex/raw/tma2025/clustering/${sample_name}.h5ad" \
--output_dir "/mnt/radonc-li02/public/he_codex/raw/tma2025/clustering/${sample_name}" 
