"""How a cellpose mask was produced, recorded in the mask's own TIFF header.

Dense cores can crash cellpose's flow_threshold QC step, so step 5 may fall back to a
degraded run or, failing that, write an all-zero placeholder. Readers need to tell those
apart from a normal mask, and this is the shared vocabulary for doing so.

Status is embedded in the TIFF rather than kept in a separate file so that pixels and
status commit together in a single atomic rename - there is no window in which a mask and
a file describing it can disagree. The .error.json / .salvaged.json sidecars written
beside a mask are a human-readable trail, not the authority.

This module deliberately imports only tifffile: step 5 runs in the cellpose env and
step 7 in sp_process, which has no torch, so the two cannot share cellpose_detection.py.
"""
import tifffile

STATUS_OK = "ok"                  # segmented normally
STATUS_SALVAGED = "salvaged"      # segmented with the flow_threshold QC step disabled
STATUS_ERROR = "error"            # all-zero placeholder; segmentation failed
STATUS_UNREADABLE = "unreadable"  # header could not be parsed (returned, never written)


def read_mask_meta(mask_path):
    """Return a mask's TIFF header metadata, or None if it cannot be read.

    Reads the IFD only - a few KB even on a 400 MB mask - so callers can decide what to
    do with a core before paying to load its pixels. A mask written before this metadata
    existed returns a dict with no "sp_" keys.
    """
    try:
        with tifffile.TiffFile(mask_path) as tif:
            return (tif.shaped_metadata or [{}])[0] or {}
    except Exception:
        return None


def read_mask_status(mask_path):
    """Return (status, n_cells) from a mask's TIFF header, without reading any pixels.

    status is None for masks written before this metadata existed, so callers can tell
    "produced by an older run" apart from "produced by a failed run". n_cells is None
    whenever status is.
    """
    meta = read_mask_meta(mask_path)
    if meta is None:
        return STATUS_UNREADABLE, None
    return meta.get("sp_status"), meta.get("sp_n_cells")


def sidecar_path(mask_path, kind):
    """Path of a mask's "error"/"salvaged" sidecar.

    Named so it sorts after the .tif it describes, because sp_marker_stats._find_tissue_mask
    globs f"{stem}*" and takes the first hit - a name sorting before the mask would hand
    that helper a JSON file to imread.
    """
    return f"{mask_path}.{kind}.json"
