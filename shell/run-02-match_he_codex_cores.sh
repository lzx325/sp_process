set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "$PROJECT_DIR/shell/setup_conda.sh" "/mnt/radonc-li02/private/lzx325/env/scimap2"

if true; then
    data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"
    he_dir="$data_dir/TA-29/HE_cores"
    codex_dir="$data_dir/TA-29/CODEX_cores"
    he_out_dir="$data_dir/TA-29/HE_cores_matched"
    codex_out_dir="$data_dir/TA-29/CODEX_cores_matched"
    mapping_csv="$data_dir/raw_scan/project/external_data/TA-29_metadata.csv"

    python "$PROJECT_DIR/src/match_he_codex_cores.py" \
        --he-dir "$he_dir" \
        --codex-dir "$codex_dir" \
        --he-out-dir "$he_out_dir" \
        --codex-out-dir "$codex_out_dir" \
        --mapping-csv "$mapping_csv"
fi

if false; then
tma_names=("TA-70")
    for tma_name in "${tma_names[@]}"; do
        echo "$tma_name"
        he_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/HE_${tma_name}_cores"
        codex_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_${tma_name}_cores"
        he_out_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/HE_${tma_name}_cores_matched"
        codex_out_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_${tma_name}_cores_matched"
        mapping_csv="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/${tma_name}_metadata.csv"

        python "$PROJECT_DIR/src/match_he_codex_cores.py" \
            --he-dir "$he_dir" \
            --codex-dir "$codex_dir" \
            --he-out-dir "$he_out_dir" \
            --codex-out-dir "$codex_out_dir" \
            --mapping-csv "$mapping_csv"
    done
fi

if false; then
    he_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_1046553_cores/"
    codex_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/IF_1046553_cores/"
    he_out_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_1046553_cores_matched/"
    codex_out_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/IF_1046553_cores_matched/"
    mapping_csv="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/1046553_metadata.csv"

    python "$PROJECT_DIR/src/match_he_codex_cores.py" \
        --he-dir "$he_dir" \
        --codex-dir "$codex_dir" \
        --he-out-dir "$he_out_dir" \
        --codex-out-dir "$codex_out_dir" \
        --mapping-csv "$mapping_csv"
fi
