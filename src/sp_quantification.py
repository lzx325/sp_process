import json
from typing import NamedTuple, Optional
import numpy as np
import pandas as pd
import h5py
import anndata as ad
from anndata.experimental import concat_on_disk
from anndata.io import read_elem, write_elem
from tqdm import trange, tqdm
from pathlib import Path
import tifffile
from loguru import logger
from multiprocessing import Pool
from tif_utils import open_wsi
from sp_marker_stats import _parse_channel_list, _parse_rename_map
from mask_status import STATUS_ERROR, STATUS_OK, read_mask_meta, sidecar_path


class CoreResult(NamedTuple):
    """Outcome of one core. core_fp is None exactly when the core was excluded.

    sample is carried explicitly rather than read back off core_fp because excluded cores
    have no path, and they still need to be grouped by slide and reported by name.
    """
    sample: str
    core_fp: Optional[Path]
    reason: Optional[str] = None  # set iff core_fp is None
    note: Optional[str] = None    # mask status of an included core, e.g. "salvaged"

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

    # Labels are assumed to run 1..max; a gap divides by zero below, and the errstate
    # guard would hide it, so say so rather than emitting silent NaN rows.
    n_empty = int((pixel_counts == 0).sum())
    if n_empty:
        logger.warning(f"{n_empty}/{n_cells} labels have no pixels; their rows will be NaN")

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


def resolve_channel_names(image_name, metadata_names, n_channels, force_channel_names=None, rename_map=None):
    """
    Channel names to save for one image: the forced names if given, otherwise the metadata names
    (generic names if there are none) with rename_map applied. Raises ValueError if the forced names
    don't match the channel count or the renames produce duplicates.
    """
    if force_channel_names is not None:
        if len(force_channel_names) != n_channels:
            raise ValueError(
                f"'{image_name}' has {n_channels} channels but "
                f"{len(force_channel_names)} names were forced"
            )
        return list(force_channel_names)

    channel_names = metadata_names
    if channel_names is None:
        channel_names = [f"Channel_{i}" for i in range(n_channels)]
        logger.warning(f"No channel names in '{image_name}'; using generic names: {channel_names}")
    if rename_map is not None:
        channel_names = [rename_map.get(ch, ch) for ch in channel_names]
        collisions = sorted({n for n in channel_names if channel_names.count(n) > 1})
        if collisions:
            raise ValueError(f"--rename_channel_names produces duplicate channel names in '{image_name}': {collisions}")
    return list(channel_names)


def read_h5ad_header(h5ad_fp):
    """(var_names, n_obs) of an .h5ad file, without loading X."""
    adata = ad.read_h5ad(h5ad_fp, backed='r')
    try:
        return adata.var_names.tolist(), adata.shape[0]
    finally:
        adata.file.close()


def attach_uns(h5ad_fp, extra):
    """Add entries to an .h5ad file's uns after the fact.

    concat_on_disk writes no uns at all, so there is normally no group to extend. The
    whole mapping is rewritten rather than grafting a child onto a bare group, because
    h5py's require_group would leave uns itself without the encoding attributes anndata
    expects when reading it back.
    """
    with h5py.File(h5ad_fp, 'a') as f:
        uns = dict(read_elem(f['uns'])) if 'uns' in f else {}
        uns.update(extra)
        if 'uns' in f:
            del f['uns']
        write_elem(f, 'uns', uns)


def _inspect_mask(mask_path):
    """(reason, meta) for one mask: reason is why it cannot be quantified, or None if it can.

    Checked before the image is opened: two stat()s and one TIFF header read, against
    ~400 MB of pixels for a core we may be about to drop anyway. meta is {} when there is
    nothing to read.
    """
    if not mask_path.exists():
        detail = ""
        err_fp = Path(sidecar_path(mask_path, "error"))
        if err_fp.exists():
            # Step 5 records a load failure here when it had no shape to write a mask with
            try:
                detail = f": {json.loads(err_fp.read_text()).get('error', '')}"
            except Exception:
                pass
        return f"mask_missing{detail}", {}

    meta = read_mask_meta(mask_path)
    if meta is None:
        return "mask_unreadable", {}
    if meta.get('sp_status') == STATUS_ERROR:
        return f"cellpose_error: {meta.get('sp_error') or 'unknown'}", meta
    if meta.get('sp_n_cells') == 0:
        return "empty_mask: 0 labels", meta
    return None, meta


def process_one_image(args):
    """Quantify one core and write it to per_core_dir. Returns a CoreResult."""
    image_file, mask_dir, per_core_dir, force_channel_names, rename_map = args
    image_file = Path(image_file)
    mask_dir = Path(mask_dir)
    sample_name = image_file.stem.split('.')[0]

    mask_path = mask_dir / f"{sample_name}.tif"
    core_fp = Path(per_core_dir) / f"{sample_name}.h5ad"

    reason, mask_meta = _inspect_mask(mask_path)
    if reason is not None:
        return CoreResult(sample_name, None, reason)
    # Both are None on masks written before this metadata existed, which is why the
    # post-read n_cells check below still has to exist.
    mask_status = mask_meta.get('sp_status')
    n_cells_hdr = mask_meta.get('sp_n_cells')

    try:
        tif, arr, metadata = open_wsi(image_file)
        try:
            channel_names = resolve_channel_names(
                image_file.name,
                metadata.get('channel_names'),
                arr.shape[0],
                force_channel_names=force_channel_names,
                rename_map=rename_map,
            )
            # Reuse a core from an earlier run only if it was saved with the same channel
            # names AND holds as many cells as the mask now claims. Matching names alone
            # is not enough: a core quantified from a mask that has since been fixed would
            # otherwise be served from the stale file forever, without the mask being read.
            if core_fp.exists():
                prior_names, prior_n_obs = read_h5ad_header(core_fp)
                if prior_names == channel_names and (n_cells_hdr is None or prior_n_obs == n_cells_hdr):
                    return CoreResult(sample_name, core_fp, None, note=mask_status)
                logger.info(f"{core_fp} does not match this run (channel names or cell count), "
                            f"re-quantifying {sample_name}.")

            mask = tifffile.imread(mask_path)

            n_cells = int(np.max(mask))
            if n_cells == 0:
                # Reachable for masks with no sp_n_cells to check up front
                return CoreResult(sample_name, None, "empty_mask: 0 labels")
            cell_ids = [f"{i}--{sample_name}" for i in range(1, n_cells + 1)]

            adata = aggregate_cell_marker_expressions(
                arr,
                mask,
                cell_ids=cell_ids,
                channel_names=channel_names
            )
        finally:
            tif.close()
        adata.obs['sample'] = sample_name
        adata.obs['slide'] = get_slide_name(sample_name)
        # Plain str, not categorical: uns is dropped by concat and subsetting, so the only
        # place a per-cell QC flag survives is obs. "legacy" means a mask from before
        # step 5 started recording this.
        adata.obs['mask_qc'] = mask_status or 'legacy'

        # Write then rename so an interrupted run never leaves a truncated file that gets reused
        tmp_fp = core_fp.with_suffix('.h5ad.tmp')
        adata.write(tmp_fp)
        tmp_fp.replace(core_fp)
        return CoreResult(sample_name, core_fp, None, note=mask_status)

    except Exception as e:
        logger.error(f"Error processing {sample_name}: {e}, skipping.")
        return CoreResult(sample_name, None, f"exception: {type(e).__name__}: {e}")


def quantify_sp_images(input_image_dir, mask_dir, output_fp, n_workers=None, slide_split=False,
                       force_channel_names=None, rename_map=None):
    """
    Quantify spatial proteomics images and save as concatenated AnnData.

    Each core is quantified independently and written to <output_stem>_per_core/<sample>.h5ad;
    cores already present there with the same channel names and cell count are reused, so an
    interrupted run can be resumed. The per-core files are then concatenated on disk, without
    loading them all into memory.

    Cores whose mask is missing, unreadable, an all-zero placeholder from a failed segmentation,
    or simply label-free are excluded. Each exclusion is logged, listed in
    <output_stem>.excluded_cores.json, and recorded in the output's uns['excluded_cores'].

    Args:
        input_image_dir (str or Path): Directory containing SP images (.ome.tiff, .qptiff, .tif, .tiff)
        mask_dir (str or Path): Directory containing segmentation masks (.tiff, .tif)
        output_fp (str or Path): Output file path for the concatenated .h5ad file
        n_workers (int, optional): Number of worker processes. Defaults to os.cpu_count().
        slide_split (bool): If True, save one AnnData file per slide parsed from sample names.
        force_channel_names (list of str, optional): Channel names overriding the image metadata.
            Every image must have exactly this many channels.
        rename_map (dict, optional): {old: new} renames applied to the metadata channel names.
            Old names that are not channels are ignored.
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

    # Check forced/renamed channel names against every image before any heavy work
    if force_channel_names is not None or rename_map is not None:
        errors = []
        for image_file in image_files:
            try:
                tif, arr, metadata = open_wsi(image_file)
                tif.close()
                resolve_channel_names(image_file.name, metadata.get('channel_names'), arr.shape[0],
                                      force_channel_names=force_channel_names, rename_map=rename_map)
            except ValueError as e:
                errors.append(str(e))
        if errors:
            details = "\n".join(f"  {e}" for e in errors)
            raise ValueError(f"Channel names cannot be applied to {len(errors)} image(s):\n{details}")

    task_args = [(image_file, mask_dir, per_core_dir, force_channel_names, rename_map) for image_file in image_files]
    # task_args = [(image_file, mask_dir, per_core_dir, force_channel_names, rename_map) for image_file in image_files if image_file.name=='TA-150_12-1.IF.ome.tif'] # for debug
    logger.info(f"Processing {len(task_args)} images, per-core outputs in {per_core_dir}")

    with Pool(processes=n_workers) as pool:
        results = list(tqdm(pool.imap(process_one_image, task_args), total=len(task_args), desc="Quantifying images"))

    core_fps = [r.core_fp for r in results if r.core_fp is not None]
    excluded = {r.sample: r.reason for r in results if r.core_fp is None}
    degraded = {r.sample: r.note for r in results if r.core_fp is not None
                and r.note not in (None, STATUS_OK)}

    for sample, reason in sorted(excluded.items()):
        logger.warning(f"Excluded {sample}: {reason}")
    for sample, note in sorted(degraded.items()):
        logger.warning(f"Included {sample} with degraded mask: {note}")
    logger.info(f"{len(results)} images, {len(core_fps)} quantified, {len(excluded)} excluded")

    # Written before the concat, so the report survives even when nothing can be concatenated
    # and when a slide ends up with no cores to attach uns to.
    report_fp = output_fp.with_name(f"{output_fp.stem}.excluded_cores.json")
    report_fp.write_text(json.dumps({'n_total': len(results), 'n_included': len(core_fps),
                                     'excluded': excluded, 'degraded': degraded}, indent=2))
    logger.info(f"Wrote exclusion report to {report_fp}")

    if not core_fps:
        kinds = ", ".join(sorted({r.split(':')[0] for r in excluded.values()}))
        raise ValueError(f"No valid samples processed: all {len(excluded)} cores excluded "
                         f"({kinds}). See {report_fp}")

    if slide_split:
        slide_to_fps = {}
        for fp in core_fps:
            slide_to_fps.setdefault(get_slide_name(fp.stem), []).append(fp)
        for slide_name in sorted(slide_to_fps):
            slide_output_fp = output_fp.with_name(f"{output_fp.stem}_{slide_name}{output_fp.suffix}")
            concat_on_disk(slide_to_fps[slide_name], slide_output_fp, join='outer')
            attach_uns(slide_output_fp, {
                'excluded_cores': {k: v for k, v in excluded.items() if get_slide_name(k) == slide_name},
                'degraded_cores': {k: v for k, v in degraded.items() if get_slide_name(k) == slide_name},
            })
            logger.info(f"Saved slide AnnData for {slide_name} ({len(slide_to_fps[slide_name])} cores) to {slide_output_fp}")
    else:
        concat_on_disk(core_fps, output_fp, join='outer')
        # Always written, even when empty, so "nothing excluded" is distinguishable from
        # "written by a version that did not record this".
        attach_uns(output_fp, {'excluded_cores': excluded, 'degraded_cores': degraded})
        logger.info(f"Saved concatenated AnnData to {output_fp}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Quantify spatial proteomics images.")
    parser.add_argument("--input_image_dir", required=True, help="Directory containing SP images")
    parser.add_argument("--mask_dir", required=True, help="Directory containing segmentation masks")
    parser.add_argument("--output_fp", required=True, help="Output .h5ad file path")
    parser.add_argument("--n_workers", type=int, default=None, help="Number of worker processes (default: cpu count)")
    parser.add_argument("--slide-split", action="store_true", help="Split output AnnData by slide name parsed from sample names")
    name_group = parser.add_mutually_exclusive_group()
    name_group.add_argument(
        "--force_channel_names", default=None,
        help="Comma-separated channel names (e.g. 'DAPI,CD3,CD8') overriding the image metadata. "
             "Every image must have exactly this many channels."
    )
    name_group.add_argument(
        "--rename_channel_names", default=None,
        help="Comma-separated <old>:<new> pairs (e.g. 'DAPI:Nuclei,CD3e:CD3') applied to channel names "
             "in the saved AnnData. Pairs whose <old> name is not a channel are ignored."
    )
    args = parser.parse_args()

    force_channel_names = None
    rename_map = None
    try:
        if args.force_channel_names is not None:
            force_channel_names = _parse_channel_list(args.force_channel_names)
        if args.rename_channel_names is not None:
            rename_map = _parse_rename_map(args.rename_channel_names)
    except ValueError as exc:
        parser.error(str(exc))

    quantify_sp_images(args.input_image_dir, args.mask_dir, args.output_fp, n_workers=args.n_workers, slide_split=args.slide_split,
                       force_channel_names=force_channel_names, rename_map=rename_map)
