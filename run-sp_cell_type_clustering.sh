set -e
PROJECT_DIR="/mnt/radonc-li02/private/lzx325/virtual-biomarking"
source "$PROJECT_DIR/shell_scripts/setup_conda_py.sh" "scimap2"

sample_name="$1"
python -u "$PROJECT_DIR/tools/preprocess/he_codex/6_1_sp_other/sp_cell_type_clustering.py" \
--input_anndata_fp "/mnt/radonc-li02/public/he_codex/raw/tma2025/clustering/${sample_name}.h5ad" \
--output_dir "/mnt/radonc-li02/public/he_codex/raw/tma2025/clustering/${sample_name}" 
