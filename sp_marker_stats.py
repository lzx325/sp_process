import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import tifffile
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from tif_utils import open_wsi


def _find_tissue_mask(wsi_path: Path, mask_dir: Path):
    """Return first mask file in mask_dir whose name starts with wsi_path's prefix (before first dot), or None."""
    stem = wsi_path.name.split(".")[0]
    candidates = sorted(mask_dir.glob(f"{stem}*"))
    if candidates:
        return candidates[0]
    return None


def _compute_channel_stats(channel_arr: np.ndarray, mask=None, max_pixels=None):
    """
    Compute summary statistics for a single 2-D channel array (H, W).

    Parameters
    ----------
    channel_arr : np.ndarray, shape (H, W)
    mask        : np.ndarray or None — if provided, only pixels where mask > 0 are used
    max_pixels  : int or None — if set, randomly subsample this many pixels

    Returns
    -------
    dict with keys: min, max, mean, std, p1, p99, p99_9, sampled_pixel_count
    """
    if mask is not None:
        pixels = channel_arr[mask > 0].astype(np.float64)
    else:
        pixels = channel_arr.ravel().astype(np.float64)

    if max_pixels is not None and len(pixels) > max_pixels:
        idx = np.random.choice(len(pixels), max_pixels, replace=False)
        pixels = pixels[idx]

    return {
        "min":                 float(pixels.min()),
        "max":                 float(pixels.max()),
        "mean":                float(pixels.mean()),
        "std":                 float(pixels.std()),
        "p1":                  float(np.percentile(pixels, 1)),
        "p99":                 float(np.percentile(pixels, 99)),
        "p99_9":               float(np.percentile(pixels, 99.9)),
        "sampled_pixel_count": int(len(pixels)),
    }


def _process_wsi(wsi_path: Path, tissue_mask_dir, max_pixels):
    """Process one WSI file. Returns (stem, channel_names, file_stats). Raises on failure."""
    mask = None
    if tissue_mask_dir is not None:
        mask_path = _find_tissue_mask(wsi_path, tissue_mask_dir)
        if mask_path is None:
            logger.warning(f"Tissue mask not found for '{wsi_path.name}' in {tissue_mask_dir}")
        else:
            mask = tifffile.imread(mask_path)

    tif, arr, metadata = open_wsi(wsi_path)

    channel_names = metadata.get("channel_names")
    if channel_names is None:
        channel_names = [f"Channel_{i}" for i in range(arr.shape[0])]
        logger.warning(f"No channel names in '{wsi_path.name}'; using generic names: {channel_names}")

    file_stats = {}
    for c_idx, ch_name in enumerate(channel_names):
        channel_arr = arr[c_idx].compute()
        file_stats[ch_name] = _compute_channel_stats(channel_arr, mask=mask, max_pixels=max_pixels)

    tif.close()
    return wsi_path.name.split(".")[0], channel_names, file_stats


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-channel marker statistics from multiplex spatial proteomics WSI files."
    )
    parser.add_argument(
        "--input_dir", required=True,
        help="Directory containing WSI files (.qptiff, .ome.tiff, .tif, .tiff, .svs)."
    )
    parser.add_argument(
        "--output_fp", required=True,
        help="Output JSON file path for per-sample per-channel statistics."
    )
    parser.add_argument(
        "--tissue_mask_dir", default=None,
        help="Optional directory of tissue mask TIFFs. Matched to each WSI by file-stem prefix."
    )
    parser.add_argument(
        "--max_pixels", type=int, default=None,
        help="If set, randomly sample at most this many pixels per channel per file (reduces memory use)."
    )
    parser.add_argument(
        "--n_workers", type=int, default=1,
        help="Number of parallel worker processes (default: 1)."
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_fp = Path(args.output_fp)
    tissue_mask_dir = Path(args.tissue_mask_dir) if args.tissue_mask_dir else None

    # Discover WSI files
    wsi_extensions = [".qptiff", ".tif", ".tiff", ".svs"]
    wsi_files = sorted({p for ext in wsi_extensions for p in input_dir.glob(f"*{ext}")})

    if not wsi_files:
        logger.error(f"No WSI files found in {input_dir}")
        sys.exit(1)

    logger.info(f"Found {len(wsi_files)} WSI file(s) in {input_dir}")

    # --- Parallel processing ---
    raw_results = {}  # wsi_path -> (stem, channel_names, file_stats)
    with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
        futures = {
            executor.submit(_process_wsi, wsi_path, tissue_mask_dir, args.max_pixels): wsi_path
            for wsi_path in wsi_files
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="WSI files"):
            wsi_path = futures[future]
            try:
                raw_results[wsi_path] = future.result()
                logger.info(f"Done: {wsi_path.name}")
            except Exception as exc:
                logger.error(f"Failed to process '{wsi_path.name}': {exc}")

    # --- Consistency check (in original sorted order) and collect stats ---
    all_stats = {}
    reference_channel_names = None
    for wsi_path in wsi_files:
        if wsi_path not in raw_results:
            continue
        stem, channel_names, file_stats = raw_results[wsi_path]
        if reference_channel_names is None:
            reference_channel_names = channel_names
        elif channel_names != reference_channel_names:
            logger.warning(
                f"Channel names in '{wsi_path.name}' differ from the reference set.\n"
                f"  Reference : {reference_channel_names}\n"
                f"  This file : {channel_names}"
            )
        all_stats[stem] = file_stats

    # --- Average statistics across all slides ---
    # Collect per-channel lists of each stat field, then average
    averaged = {}
    scalar_fields = ["min", "max", "mean", "std", "p1", "p99", "p99_9"]
    # Preserve original channel order from the reference slide; append any extras from inconsistent slides
    extra = [ch for fs in all_stats.values() for ch in fs if ch not in (reference_channel_names or [])]
    seen = set()
    all_channel_names = []
    for ch in (reference_channel_names or []) + extra:
        if ch not in seen:
            seen.add(ch)
            all_channel_names.append(ch)
    for ch_name in all_channel_names:
        per_slide = [fs[ch_name] for fs in all_stats.values() if ch_name in fs]
        if not per_slide:
            continue
        averaged[ch_name] = {
            field: float(np.mean([s[field] for s in per_slide]))
            for field in scalar_fields
        }
        averaged[ch_name]["sampled_pixel_count"] = int(sum(s["sampled_pixel_count"] for s in per_slide))

    # --- Write output ---
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    with open(output_fp, "w") as f:
        json.dump(averaged, f, indent=2)
    logger.info(f"Statistics (averaged across {len(all_stats)} slide(s)) saved to {output_fp}")


if __name__ == "__main__":
    main()
