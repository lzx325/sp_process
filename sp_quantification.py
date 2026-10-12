import numpy as np
import pandas as pd
import anndata as ad
from tqdm import trange, tqdm
from pathlib import Path
import tifffile
from loguru import logger
from multiprocessing import Pool
from tif_utils import open_wsi

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
    if codex_img.dtype == np.uint16:
        codex_img = codex_img.astype(np.float32)/65535
    elif codex_img.dtype == np.uint8:
        codex_img = codex_img.astype(np.float32)/255
    else:
        assert False, "codex_img dtype unknown"
    if cell_ids is not None:
        assert len(cell_ids) == np.max(seg_mask), "Length of cell_ids must match the number of cells in seg_mask"
        assert len(set(cell_ids)) == len(cell_ids), "cell_ids must be unique"
        
    assert tuple(seg_mask.shape) == tuple(codex_img.shape[1:])
    n_cells = np.max(seg_mask)
    n_channels = codex_img.shape[0]
    
    sp_exp_mat = np.zeros((n_cells, n_channels))
    for cell_index in range(1,n_cells+1):
        row_indices, col_indices = np.where(seg_mask == cell_index)
        sp_exp_mat[cell_index-1, :] = codex_img[:, row_indices, col_indices].mean(axis=1)

    adata = ad.AnnData(X = sp_exp_mat)
    if cell_ids is None:
        cell_ids = list(range(1, n_cells+1))

    adata.obs = pd.DataFrame(index=cell_ids)
        
    if channel_names is not None:
        adata.var = pd.DataFrame(index=channel_names)
    return adata


def process_one_image(args):
    image_file, mask_dir = args
    image_file = Path(image_file)
    mask_dir = Path(mask_dir)
    sample_name = image_file.stem.split('.')[0]

    mask_path = mask_dir / f"{sample_name}.tif"

    try:
        tif, arr, metadata = open_wsi(image_file)
        channel_names = metadata.get('channel_names')
        tif.close()

        mask = tifffile.imread(mask_path)

        n_cells = int(np.max(mask))
        cell_ids = [f"{i}--{sample_name}" for i in range(1, n_cells + 1)]

        adata = aggregate_cell_marker_expressions(
            arr.compute() if hasattr(arr, 'compute') else arr,
            mask,
            cell_ids=cell_ids,
            channel_names=channel_names
        )
        adata.obs['sample'] = sample_name
        return adata

    except Exception as e:
        logger.error(f"Error processing {sample_name}: {e}, skipping.")
        return None


def quantify_sp_images(input_image_dir, mask_dir, output_fp, n_workers=None, slide_split=False):
    """
    Quantify spatial proteomics images and save as concatenated AnnData.

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

    # List image files
    image_extensions = ['.ome.tiff', '.qptiff', '.tif', '.tiff']
    image_files = []
    for ext in image_extensions:
        image_files.extend(input_image_dir.glob(f'*{ext}'))
    image_files = sorted(set(image_files))  # Remove duplicates if any
    task_args = [(image_file, mask_dir) for image_file in image_files]
    # task_args = [(image_file, mask_dir) for image_file in image_files if image_file.name=='TA-150_12-1.IF.ome.tif'] # for debug
    logger.info(f"Processing {len(task_args)} images")

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(process_one_image, task_args), total=len(task_args), desc="Quantifying images"))
    adatas = [r for r in results if r is not None]

    if not adatas:
        raise ValueError("No valid samples processed.")

    # Concatenate
    adata_combined = ad.concat(adatas, join='outer')
    
    if slide_split:
        adata_combined.obs['slide'] = adata_combined.obs['sample'].astype(str).str.split('-').str[0]
        for slide_name in sorted(adata_combined.obs['slide'].unique()):
            slide_adata = adata_combined[adata_combined.obs['slide'] == slide_name].copy()
            slide_output_fp = output_fp.with_name(f"{output_fp.stem}_{slide_name}{output_fp.suffix}")
            slide_adata.write(slide_output_fp)
            logger.info(f"Saved slide AnnData for {slide_name} to {slide_output_fp}")
    else:
        # Write output
        adata_combined.write(output_fp)
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
