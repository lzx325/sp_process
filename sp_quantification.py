import numpy as np
import pandas as pd
import anndata as ad
from anndata.experimental import concat_on_disk
from tqdm import trange, tqdm
from pathlib import Path
import tifffile
from loguru import logger
from multiprocessing import Pool
from tif_utils import open_wsi
from sp_marker_definitions import MARKER_ALIASES_TMA2025

logger.remove()
logger.add(
    lambda msg: print(msg, end=""),
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{file}:{line}</cyan> - {message}",
    colorize=True,
)

def aggregate_cell_marker_expressions(
    codex_img,
    seg_mask,
    cell_ids = None,
    channel_names = None
):
    """
    Mean intensity of each channel within each cell.

    codex_img (C, H, W) may be a numpy or dask array; it is read one channel at a
    time so only a single plane is held in memory.
    """
    if codex_img.dtype == np.uint16:
        max_val = 65535
    elif codex_img.dtype == np.uint8:
        max_val = 255
    else:
        assert False, "codex_img dtype unknown"
    if cell_ids is not None:
        assert len(cell_ids) == np.max(seg_mask), "Length of cell_ids must match the number of cells in seg_mask"
        assert len(set(cell_ids)) == len(cell_ids), "cell_ids must be unique"

    assert tuple(seg_mask.shape) == tuple(codex_img.shape[1:])
    n_cells = int(np.max(seg_mask))
    n_channels = codex_img.shape[0]

    # Only labeled pixels contribute, so index them once and reuse for every channel
    fg_idx = np.flatnonzero(seg_mask)
    fg_labels = seg_mask.ravel()[fg_idx]
    pixel_counts = np.bincount(fg_labels, minlength=n_cells + 1)[1:]

    sp_exp_mat = np.zeros((n_cells, n_channels))
    for c in range(n_channels):
        channel = codex_img[c]
        # Parallelism comes from the per-core worker pool; dask's thread pool on top only adds lock contention
        channel = channel.compute(scheduler='synchronous') if hasattr(channel, 'compute') else np.asarray(channel)
        sums = np.bincount(fg_labels, weights=channel.ravel()[fg_idx], minlength=n_cells + 1)[1:]
        with np.errstate(invalid='ignore', divide='ignore'):
            sp_exp_mat[:, c] = sums / pixel_counts / max_val
        del channel

    adata = ad.AnnData(X = sp_exp_mat)
    if cell_ids is None:
        cell_ids = list(range(1, n_cells+1))

    adata.obs = pd.DataFrame(index=cell_ids)
        
    if channel_names is not None:
        adata.var = pd.DataFrame(index=channel_names)
    return adata


def get_slide_name(sample_name):
    """'TA-237A_10-10' -> 'TA-237A' (the core position follows the last underscore)."""
    return sample_name.rsplit('_', 1)[0]


def process_one_image(args):
    """Quantify one core and write it to per_core_dir. Returns the written path, or None on failure."""
    image_file, mask_dir, per_core_dir = args
    image_file = Path(image_file)
    mask_dir = Path(mask_dir)
    sample_name = image_file.stem.split('.')[0]

    mask_path = mask_dir / f"{sample_name}.tif"
    core_fp = Path(per_core_dir) / f"{sample_name}.h5ad"
    if core_fp.exists():
        return core_fp

    try:
        mask = tifffile.imread(mask_path)

        n_cells = int(np.max(mask))
        cell_ids = [f"{i}--{sample_name}" for i in range(1, n_cells + 1)]

        tif, arr, metadata = open_wsi(image_file)
        try:
            adata = aggregate_cell_marker_expressions(
                arr,
                mask,
                cell_ids=cell_ids,
                channel_names=metadata.get('channel_names')
            )
        finally:
            tif.close()
        adata.var_names = [MARKER_ALIASES_TMA2025.get(m, m) for m in adata.var_names]
        adata.obs['sample'] = sample_name
        adata.obs['slide'] = get_slide_name(sample_name)

        # Write then rename so an interrupted run never leaves a truncated file that gets reused
        tmp_fp = core_fp.with_suffix('.h5ad.tmp')
        adata.write(tmp_fp)
        tmp_fp.rename(core_fp)
        return core_fp

    except Exception as e:
        logger.error(f"Error processing {sample_name}: {e}, skipping.")
        return None


def quantify_sp_images(input_image_dir, mask_dir, output_fp, n_workers=None, slide_split=False):
    """
    Quantify spatial proteomics images and save as concatenated AnnData.

    Each core is quantified independently and written to <output_stem>_per_core/<sample>.h5ad;
    cores already present there are skipped, so an interrupted run can be resumed. The per-core
    files are then concatenated on disk, without loading them all into memory.

    Args:
        input_image_dir (str or Path): Directory containing SP images (.ome.tiff, .qptiff, .tif, .tiff)
        mask_dir (str or Path): Directory containing segmentation masks (.tiff, .tif)
        output_fp (str or Path): Output file path for the concatenated .h5ad file
        n_workers (int, optional): Number of worker processes. Defaults to os.cpu_count().
        slide_split (bool): If True, save one AnnData file per slide parsed from sample names.
    """
    input_image_dir = Path(input_image_dir)
    mask_dir = Path(mask_dir)
    output_fp = Path(output_fp)
    output_fp.parent.mkdir(parents=True, exist_ok=True)
    per_core_dir = output_fp.with_name(f"{output_fp.stem}_per_core")
    per_core_dir.mkdir(exist_ok=True)

    # List image files
    image_extensions = ['.ome.tiff', '.qptiff', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_image_dir.glob(f'*{ext}'))
    image_files = sorted(set(image_files))  # Remove duplicates if any
    task_args = [(image_file, mask_dir, per_core_dir) for image_file in image_files]
    # task_args = [(image_file, mask_dir, per_core_dir) for image_file in image_files if image_file.name=='TA-150_12-1.IF.ome.tif'] # for debug
    logger.info(f"Processing {len(task_args)} images, per-core outputs in {per_core_dir}")

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(process_one_image, task_args), total=len(task_args), desc="Quantifying images"))
    core_fps = [r for r in results if r is not None]

    if not core_fps:
        raise ValueError("No valid samples processed.")

    if slide_split:
        slide_to_fps = {}
        for fp in core_fps:
            slide_to_fps.setdefault(get_slide_name(fp.stem), []).append(fp)
        for slide_name in sorted(slide_to_fps):
            slide_output_fp = output_fp.with_name(f"{output_fp.stem}_{slide_name}{output_fp.suffix}")
            concat_on_disk(slide_to_fps[slide_name], slide_output_fp, join='outer')
            logger.info(f"Saved slide AnnData for {slide_name} ({len(slide_to_fps[slide_name])} cores) to {slide_output_fp}")
    else:
        concat_on_disk(core_fps, output_fp, join='outer')
        logger.info(f"Saved concatenated AnnData to {output_fp}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantify spatial proteomics images.")
    parser.add_argument("--input_image_dir", required=True, help="Directory containing SP images")
    parser.add_argument("--mask_dir", required=True, help="Directory containing segmentation masks")
    parser.add_argument("--output_fp", required=True, help="Output .h5ad file path")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of worker processes (default: cpu count)")
    parser.add_argument("--slide-split", action="store_true", help="Split output AnnData by slide name parsed from sample names")
    args = parser.parse_args()

    quantify_sp_images(args.input_image_dir, args.mask_dir, args.output_fp, n_workers=args.n_workers, slide_split=args.slide_split)
