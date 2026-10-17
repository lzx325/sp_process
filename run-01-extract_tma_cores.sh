set -e
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$PROJECT_DIR/setup_conda.sh" "/mnt/radonc-li02/private/lzx325/env/scimap2"

if true; then

tma_names=(TA-29)
diameters=(800)
data_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_test"
for i in "${!tma_names[@]}"; do
    tma_name=${tma_names[$i]}
    in_wsi_fp="$data_dir/raw_scan/HE/${tma_name}.svs"
    tma_fp="$data_dir/raw_scan/project/external_data/HE_${tma_name}.tsv"
    out_dir="$data_dir/tma_cores/HE_${tma_name}_cores"
    python "$PROJECT_DIR/extract_tma_cores.py" \
        "$in_wsi_fp" \
        "$tma_fp" \
        "$out_dir" \
        --core-diameter-um ${diameters[$i]}
    
    in_wsi_fp="$data_dir/raw_scan/CODEX/${tma_name}_Scan1.qptiff"
    tma_fp="$data_dir/raw_scan/project/external_data/CODEX_${tma_name}.tsv"
    out_dir="$data_dir/tma_cores/CODEX_${tma_name}_cores"
    python "$PROJECT_DIR/extract_tma_cores.py" \
        "$in_wsi_fp" \
        "$tma_fp" \
        "$out_dir" \
        --core-diameter-um ${diameters[$i]}
done

fi

if false; then
tma_names=("TA-29" "TA-47" "TA-70" "TA-237A" "TA-256" "TA-257" "TA-258")
diameters=(800 800 2300 1400 1400 1300 1400)

for i in "${!tma_names[@]}"; do
    tma_name=${tma_names[$i]}
    in_wsi_fp="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/${tma_name}.svs"
    tma_fp="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_${tma_name}.tsv"
    out_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_${tma_name}_cores"
    python extract_tma_cores.py \
        "$in_wsi_fp" \
        "$tma_fp" \
        "$out_dir" \
        --core-diameter-um ${diameters[$i]} &
done
fi

if false; then
tma_name="1046553"
if false; then
    in_wsi_fp="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/HE/${tma_name}.svs"
    tma_fp="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_${tma_name}.tsv"
    out_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/HE_${tma_name}_cores"
    diameter=1400
fi
if false; then
    in_wsi_fp="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/Xenium/morphology_focus/morphology_focus_0000.ome.tif"
    tma_fp="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/IF_1046553.tsv"
    out_dir="/mnt/radonc-li02/public/he_codex/raw/Jimpi/raw_scan/project/external_data/IF_1046553_cores"
    diameter=1400
fi
python extract_tma_cores.py \
    "$in_wsi_fp" \
    "$tma_fp" \
    "$out_dir" \
    --core-diameter-um ${diameter}
fi


if false; then
tma_keys=("TA-29" "TA-47" "TA-70" "TA-237A" "TA-256" "TA-257" "TA-258" "TA-417")
tma_names=("TA-29_Scan1" "TA-47_Scan1" "GI TA-70_Scan1" "GI TA-237A_Scan1" "GI TA-256_Scan1" "GI-TA-257_Scan1" "GI TA-258_Scan1" "BRCA TA-417_Scan2")
diameters=(800 800 2100 1200 1400 1300 1250 1300)
for i in "${!tma_keys[@]}"; do
    tma_key=${tma_keys[$i]}
    tma_name=${tma_names[$i]}
    in_wsi_fp="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/CODEX/${tma_name}.qptiff"
    tma_fp="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_${tma_key}.tsv"
    out_dir="/mnt/radonc-li02/public/he_codex/raw/tma2025_3/raw_scan/project/external_data/CODEX_${tma_key}_cores"
    python extract_tma_cores.py \
        "$in_wsi_fp" \
        "$tma_fp" \
        "$out_dir" \
        --core-diameter-um ${diameters[$i]} &
done
fi