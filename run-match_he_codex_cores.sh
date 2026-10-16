set -e
source "$HOME/anaconda3/etc/profile.d/conda.sh"
conda deactivate
conda activate "/mnt/radonc-li02/private/lzx325/env/scimap2"

if false; then
tma_names=("TA-70")
    for tma_name in "${tma_names[@]}"; do
        echo "$tma_name"
        he_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/HE_${tma_name}_cores"
        codex_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_${tma_name}_cores"
        he_out_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/HE_${tma_name}_cores_matched"
        codex_out_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_${tma_name}_cores_matched"
        mapping_csv="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/${tma_name}_metadata.csv"

        SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
        python "$SCRIPT_DIR/match_he_codex_cores.py" \
            --he-dir "$he_dir" \
            --codex-dir "$codex_dir" \
            --he-out-dir "$he_out_dir" \
            --codex-out-dir "$codex_out_dir" \
            --mapping-csv "$mapping_csv"
    done
fi

if true; then
    he_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_1046553_cores/"
    codex_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/IF_1046553_cores/"
    he_out_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_1046553_cores_matched/"
    codex_out_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/IF_1046553_cores_matched/"
    mapping_csv="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/1046553_metadata.csv"

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    python "$SCRIPT_DIR/match_he_codex_cores.py" \
        --he-dir "$he_dir" \
        --codex-dir "$codex_dir" \
        --he-out-dir "$he_out_dir" \
        --codex-out-dir "$codex_out_dir" \
        --mapping-csv "$mapping_csv"
fi
