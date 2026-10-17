set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/setup_conda.sh" "/mnt/radonc-li02/private/lzx325/env/scimap2"

if true; then
    tma_name="TA-29"
    data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"
    for fp in $data_dir/tma_cores/CODEX_${tma_name}_cores_matched/*.IF.ome.tif; do
        fn=$(basename "$fp")
        core_name=${fn%.IF.ome.tif}
        codex_fp="$data_dir/tma_cores/CODEX_${tma_name}_cores_matched/${core_name}.IF.ome.tif"
        he_fp="$data_dir/tma_cores/HE_${tma_name}_cores_matched/${core_name}.HE.ome.tif"
        output_dir="$data_dir/tma_cores/${tma_name}_registration"
        printf '%s\t%s\t%s\n' "$codex_fp" "$he_fp" "$output_dir"
    done | parallel --line-buffer -j 1 --colsep '\t' python "$PROJECT_DIR/codex_registeration_batch.py" {1} {2} {3}
fi

if false; then
    tma_names=("TA-258")
    for tma_name in "${tma_names[@]}"; do
        for fp in /mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_${tma_name}_cores_matched/*.IF.ome.tif; do
            fn=$(basename "$fp")
            core_name=${fn%.IF.ome.tif}
            codex_fp="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_${tma_name}_cores_matched/${core_name}.IF.ome.tif"
            he_fp="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/HE_${tma_name}_cores_matched/${core_name}.HE.ome.tif"
            output_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/${tma_name}_registration"
            printf '%s\t%s\t%s\n' "$codex_fp" "$he_fp" "$output_dir"
        done
    done | parallel --line-buffer -j 1 --colsep '\t' python codex_registeration_batch.py {1} {2} {3}
fi

if false; then
    tma_name="1046553"
    for fp in /mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/IF_${tma_name}_cores_matched/*.IF.ome.tif; do
        fn=$(basename "$fp")
        core_name=${fn%.IF.ome.tif}
        codex_fp="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/IF_${tma_name}_cores_matched/${core_name}.IF.ome.tif"
        he_fp="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_${tma_name}_cores_matched/${core_name}.HE.ome.tif"
        output_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/${tma_name}_registration"
        printf '%s\t%s\t%s\n' "$codex_fp" "$he_fp" "$output_dir"
    done | parallel --line-buffer -j 1 --colsep '\t' python codex_registeration_batch.py {1} {2} {3} --align-he-to-codex
fi

if false; then
    python codex_registeration_batch.py \
        "/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/IF_1046553_cores_matched/5-1.IF.ome.tif" \
        "/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_1046553_cores_matched/5-1.HE.ome.tif" \
        "/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/1046553_registration" \
        --align-he-to-codex
fi

if false; then
    python test_zarr.py
fi