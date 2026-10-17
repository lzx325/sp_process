"""
Extract TMA cores from a whole-slide image using QuPath TMA array export.

Supports:
- HE images: .svs
- Multiplex spatial proteomics: .qptiff (and .ome.tiff / .ome.tif)

Output: one OME-TIFF file per core in the specified output directory.
Uses palom.pyramid.write_pyramid for efficient pyramidal writing.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import dask.array as da
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm


from tif_utils import open_wsi, write_ome_tiff


# ---------------------------------------------------------------------------
# TMA CSV parsing (QuPath export)
# ---------------------------------------------------------------------------

def _find_column(candidates: list[str], columns: pd.Index) -> str | None:
    """Return first column name that matches any candidate (case-insensitive)."""
    cols_lower = {c.lower(): c for c in columns}
    for cand in candidates:
        key = cand.lower().strip()
        if key in cols_lower:
            return cols_lower[key]
        for c in columns:
            if c.lower().strip() == key:
                return c
    return None


def _infer_tma_separator(path: Path) -> str:
    """Infer CSV vs TSV separator from file extension. Default to comma."""
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return "\t"
    if suffix == ".csv":
        return ","
    # .txt or other: default to comma; caller could sniff if needed
    return ","


def load_tma_array(
    tma_path: str | Path,
    x_col: str | None = None,
    y_col: str | None = None,
    name_col: str | None = None,
    missing_col: str | None = None,
    coordinates_in_microns: bool = True,
    pixel_size_um: float = 0.5,
) -> pd.DataFrame:
    """
    Load TMA core positions from a QuPath-exported CSV or TSV file.

    QuPath typically exports columns such as Image, Name, Row, Column,
    and center coordinates (X, Y in microns or pixels).

    Returns a DataFrame with columns: name, x_px, y_px, missing (if present).
    """
    tma_path = Path(tma_path)
    if not tma_path.exists():
        raise FileNotFoundError(f"TMA array file not found: {tma_path}")

    sep = _infer_tma_separator(tma_path)
    df = pd.read_csv(tma_path, sep=sep)
    cols = df.columns

    # Resolve column names
    x_col = x_col or _find_column(
        ["x", "center x", "centroid x", "x_um", "x_µm", "X", "Centroid X µm"], cols
    )
    y_col = y_col or _find_column(
        ["y", "center y", "centroid y", "y_um", "y_µm", "Y", "Centroid Y µm"], cols
    )
    name_col = name_col or _find_column(
        ["name", "core", "image", "id", "core id", "row", "column"], cols
    )
    missing_col = missing_col or _find_column(
        ["missing", "is missing", "missing core"], cols
    )

    if x_col is None or y_col is None:
        raise ValueError(
            f"Could not find X and Y columns in {tma_path}. "
            f"Available columns: {list(cols)}. "
            "Use --x-col and --y-col to specify."
        )

    out = pd.DataFrame()
    out["x_um"] = pd.to_numeric(df[x_col], errors="coerce")
    out["y_um"] = pd.to_numeric(df[y_col], errors="coerce")
    if name_col:
        out["name"] = df[name_col].astype(str)
    else:
        out["name"] = [f"core_{i}" for i in range(len(df))]
    if missing_col is not None and missing_col in df.columns:
        out["missing"] = df[missing_col]
    else:
        out["missing"] = False

    # Drop rows with NaN coordinates
    out = out.dropna(subset=["x_um", "y_um"])
    if coordinates_in_microns:
        out["x_px"] = (out["x_um"] / pixel_size_um).astype(int)
        out["y_px"] = (out["y_um"] / pixel_size_um).astype(int)
    else:
        out["x_px"] = out["x_um"].astype(int)
        out["y_px"] = out["y_um"].astype(int)

    return out


# ---------------------------------------------------------------------------
# Core extraction and OME-TIFF writing
# ---------------------------------------------------------------------------

def extract_core_region(
    arr: da.Array,
    x_center_px: int,
    y_center_px: int,
    side_px: int,
) -> da.Array:
    """Extract a square region (C, H, W) centered at (x_center_px, y_center_px)."""
    _, H, W = arr.shape
    half = side_px // 2
    x0 = max(0, x_center_px - half)
    y0 = max(0, y_center_px - half)
    x1 = min(W, x0 + side_px)
    y1 = min(H, y0 + side_px)
    # Re-adjust to keep size consistent if we hit edges
    if x1 - x0 < side_px and x0 > 0:
        x0 = max(0, x1 - side_px)
    if y1 - y0 < side_px and y0 > 0:
        y0 = max(0, y1 - side_px)
    return arr[:, y0:y1, x0:x1]


def sanitize_core_name(name: str) -> str:
    """Make a safe filename from core name."""
    name = str(name).strip()
    name = re.sub(r"[^\w\-\.]+", "_", name)
    return name or "core"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract TMA cores from a WSI using QuPath TMA array export. Output: one OME-TIFF per core."
    )
    parser.add_argument(
        "wsi_path",
        type=Path,
        help="Path to the whole-slide image (.svs or .qptiff / .ome.tiff)",
    )
    parser.add_argument(
        "tma_csv",
        type=Path,
        help="Path to the TMA array file (CSV or TSV) exported from QuPath",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Directory where each TMA core will be saved as a separate OME-TIFF file",
    )
    parser.add_argument(
        "--core-diameter-um",
        type=float,
        default=800.0,
        help="Core diameter in microns (default: 1500 for 1.5 mm cores). Used to define square crop side.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in µm. If not set, read from image metadata when possible.",
    )
    parser.add_argument(
        "--magnification",
        type=float,
        default=None,
        help="Nominal magnification (e.g. 20.0). If not set, read from image metadata when possible.",
    )
    parser.add_argument(
        "--coordinates-in-microns",
        action="store_true",
        default=True,
        help="TMA CSV coordinates are in microns (default: True).",
    )
    parser.add_argument(
        "--coordinates-in-pixels",
        action="store_true",
        help="TMA CSV coordinates are in pixels (level-0).",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default=None,
        help="CSV column name for X (center). Auto-detected if not set.",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default=None,
        help="CSV column name for Y (center). Auto-detected if not set.",
    )
    parser.add_argument(
        "--name-col",
        type=str,
        default=None,
        help="CSV column name for core name. Auto-detected if not set.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        default=True,
        help="Skip cores marked as missing in the CSV (default: True).",
    )
    parser.add_argument(
        "--no-skip-missing",
        action="store_false",
        dest="skip_missing",
        help="Do not skip missing cores.",
    )
    parser.add_argument(
        "--downscale-factor",
        type=int,
        default=4,
        help="Pyramid downscale factor (default: 4).",
    )
    parser.add_argument(
        "--compression",
        type=str,
        default="zlib",
        choices=("zlib", "lzw", "deflate", "none"),
        help="TIFF compression (default: zlib).",
    )
    parser.add_argument(
        "--maxworkers",
        type=int,
        default=15,
        help="Max workers for tifffile (default: 15).",
    )
    parser.add_argument(
        "--suffix",
        type=str,
        default=".ome.tif",
        help="Output file suffix (default: .ome.tif).",
    )
    args = parser.parse_args()

    coordinates_in_microns = args.coordinates_in_microns and not args.coordinates_in_pixels
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine pixel size: from command line or from WSI metadata
    if args.pixel_size is not None:
        pixel_size_um = args.pixel_size
        logger.info(
            f"Pixel size: {pixel_size_um} µm (specified from command line)"
        )
        metadata = None
    else:
        # Open WSI first to read pixel size from metadata
        tif, arr, metadata = open_wsi(args.wsi_path)
        pixel_size_um = metadata["pixel_size"]
        logger.info(
            f"Pixel size: {pixel_size_um} µm (detected from image metadata)"
        )

    # Load TMA array (uses same pixel_size_um for µm→pixel when coordinates in microns)
    tma_df = load_tma_array(
        args.tma_csv,
        x_col=args.x_col,
        y_col=args.y_col,
        name_col=args.name_col,
        coordinates_in_microns=coordinates_in_microns,
        pixel_size_um=pixel_size_um,
    )
    if args.skip_missing and "missing" in tma_df.columns:
        tma_df = tma_df.loc[~tma_df["missing"].astype(str).str.lower().isin(("true", "1", "yes"))]
    logger.info(f"Loaded {len(tma_df)} TMA cores from {args.tma_csv}")

    # Open WSI if not already open (when pixel size was from command line)
    if args.pixel_size is not None:
        tif, arr, metadata = open_wsi(args.wsi_path)
    channel_names = metadata["channel_names"]
    # Magnification: from command line or from image metadata
    if args.magnification is not None:
        magnification = args.magnification
        logger.info(
            f"Magnification: {magnification} (specified from command line)"
        )
    else:
        magnification = metadata["magnification"]
        logger.info(
            f"Magnification: {magnification} (detected from image metadata)"
        )
    logger.info(f"WSI shape (C,H,W)={arr.shape}, pixel_size={pixel_size_um} µm")

    side_px = int(np.ceil(args.core_diameter_um / pixel_size_um))
    logger.info(f"Core crop side: {side_px} px ({args.core_diameter_um} µm)")

    try:
        for _, row in tqdm(tma_df.iterrows(), total=len(tma_df), desc="Extracting cores"):
            name = sanitize_core_name(row["name"])
            x_px = int(row["x_px"])
            y_px = int(row["y_px"])
            core_arr = extract_core_region(arr, x_px, y_px, side_px)
            out_fp = args.output_dir / f"{name}{args.suffix}"
            if out_fp.exists():
                logger.info(f"Core {name} already exists, skipping")
                continue
            write_ome_tiff(
                core_arr,
                out_fp,
                pixel_size_um=pixel_size_um,
                channel_names=channel_names,
                magnification=magnification,
                description=f"TMA core {name} from {args.wsi_path.name}",
                downscale_factor=args.downscale_factor,
                compression=args.compression,
                maxworkers=args.maxworkers,
            )
            logger.debug(f"Wrote {out_fp}")
    finally:
        tif.close()

    logger.info(f"Done. Cores written to {args.output_dir}")


if __name__ == "__main__":
    main()
