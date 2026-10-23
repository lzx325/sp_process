"""Run cellpose-SAM on the DAPI channel of each CODEX_registered OME-TIFF.

Usage:
    python cellpose_detection.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR \
        [--n_gpu N] [--dapi_channel C] [--salvage-flow-threshold T]

If --n_gpu is greater than 1, the input files are split across multiple
Cellpose worker processes, one per visible GPU.

Every aligned core gets a mask file. Dense cores can crash cellpose's flow_threshold
QC step (dynamics._extend_centers_gpu) with a CUDA kernel-launch error; those are
retried with that step disabled, and if they still fail an all-zero placeholder is
written so the mask set stays 1-to-1 with the cores. Each mask records how it was
produced in its TIFF header under "sp_status", which is what downstream reads to
decide whether to use or exclude a core.
"""
import os
import re
import json
import argparse
import glob
import multiprocessing as mp
import time
import numpy as np
import tifffile
import torch
from cellpose import models
from mask_status import (STATUS_ERROR, STATUS_OK, STATUS_SALVAGED, STATUS_UNREADABLE,
                         read_mask_status, sidecar_path)

DAPI_CHANNEL = 0
SALVAGE_FLOW_THRESHOLD = 0.0
DEFAULT_FLOW_THRESHOLD = 0.4  # cellpose's own default, recorded on normal masks

# Give up on the batch after this many cores fail back to back: that means the device or
# the process is wedged, and stamping placeholders over every remaining core would be far
# worse than failing loudly.
MAX_CONSECUTIVE_FAILURES = 3


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Directory of CODEX_registered OME-TIFF files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write cellpose masks.")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use (default: 1).")
    parser.add_argument("--dapi_channel", type=int, default=DAPI_CHANNEL,
                        help=f"Index of the DAPI channel in the OME-TIFF (default: {DAPI_CHANNEL}).")
    parser.add_argument("--salvage-flow-threshold", type=float, default=SALVAGE_FLOW_THRESHOLD,
                        help="flow_threshold to retry with when the default run fails; 0 disables "
                             f"cellpose's QC step entirely (default: {SALVAGE_FLOW_THRESHOLD}).")
    return parser.parse_args()


def _write_mask(out_mask, arr, status, n_cells, flow_threshold, error=""):
    """Write a mask and its status in one atomic step.

    Status lives in the TIFF header rather than only in a sidecar because pixels and
    status then commit together: there is no crash window in which a mask disagrees with
    a separate file describing it. The tmp name carries the pid so two processes writing
    the same output directory cannot interleave a partial write.
    """
    tmp = f"{out_mask}.{os.getpid()}.tmp"
    tifffile.imwrite(tmp, arr, compression="zlib", bigtiff=True,
                     metadata={"sp_status": status,
                               "sp_n_cells": int(n_cells),
                               "sp_flow_threshold": float(flow_threshold),
                               "sp_error": error,
                               "sp_written": time.strftime("%Y-%m-%dT%H:%M:%S")})
    os.replace(tmp, out_mask)


def _write_sidecar(path, payload):
    tmp = f"{path}.{os.getpid()}.tmp"
    with open(tmp, "w") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _clear_sidecars(out_mask):
    for kind in ("error", "salvaged"):
        try:
            os.unlink(sidecar_path(out_mask, kind))
        except FileNotFoundError:
            pass


def write_error_mask(out_mask, shape, name, worker, exc, flow_threshold):
    """Write an all-zero placeholder so every core has a mask, plus a readable sidecar.

    The sidecar goes first: it explains a mask that downstream will exclude, so it should
    be durable before that mask becomes visible. An orphan sidecar is harmless.
    """
    error = f"{type(exc).__name__}: {exc}"
    _write_sidecar(sidecar_path(out_mask, "error"),
                   {"core": name, "status": STATUS_ERROR, "error": error,
                    "flow_threshold": flow_threshold,
                    "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
    _write_mask(out_mask, np.zeros(shape, dtype=np.uint16), STATUS_ERROR, 0, flow_threshold,
                error=error)
    print(f"[error-mask] {worker} {name} wrote all-zero placeholder", flush=True)


def process_files(files, output_dir, gpu_index=None, dapi_channel=DAPI_CHANNEL,
                  salvage_flow_threshold=SALVAGE_FLOW_THRESHOLD):
    use_gpu = gpu_index is not None and torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_index}") if use_gpu else torch.device("cpu")
    worker = f"gpu={gpu_index}" if use_gpu else "cpu"
    print(f"[init] {worker} device={device} files={len(files)}", flush=True)

    model = models.CellposeModel(gpu=use_gpu, device=device)
    consecutive_failures = 0

    for f in files:
        name = re.sub(r"\.IF\.ome\.tiff?$", "", os.path.basename(f))
        out_mask = os.path.join(output_dir, f"{name}.tif")
        if os.path.exists(out_mask):
            prior, _ = read_mask_status(out_mask)
            if prior not in (STATUS_ERROR, STATUS_UNREADABLE):
                print(f"[skip] {worker} {name} status={prior or 'legacy'}", flush=True)
                continue
            # Leave the old placeholder in place until a better mask replaces it, so an
            # interrupted retry cannot turn a recorded failure into an unrecorded absence.
            print(f"[reattempt] {worker} {name} prior status={prior}", flush=True)

        t0 = time.time()
        print(f"[load] {worker} {name}", flush=True)
        try:
            with tifffile.TiffFile(f) as tif:
                dapi = tif.series[0].asarray(key=dapi_channel)
        except Exception as e:
            # No shape, so no placeholder can be written - record the reason instead of
            # dropping the core without a trace.
            print(f"[error] {worker} {name}: {e}", flush=True)
            _write_sidecar(sidecar_path(out_mask, "error"),
                           {"core": name, "status": STATUS_ERROR,
                            "error": f"image_load_failed: {type(e).__name__}: {e}",
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
            continue
        print(f"  {worker} shape={dapi.shape} dtype={dapi.dtype} load={time.time()-t0:.1f}s", flush=True)

        t1 = time.time()
        status, flow_threshold = STATUS_OK, DEFAULT_FLOW_THRESHOLD
        try:
            masks, flows, styles = model.eval(dapi)
        except Exception as e:
            # Dense cores trip a CUDA kernel-launch failure in the flow_threshold QC step
            # (cellpose dynamics._extend_centers_gpu). Retrying with that step disabled
            # never launches the offending kernel, so it still yields a usable mask.
            print(f"[salvage] {worker} {name}: {type(e).__name__}: {e}", flush=True)
            if use_gpu:
                torch.cuda.empty_cache()
            try:
                masks, flows, styles = model.eval(dapi, flow_threshold=salvage_flow_threshold)
                status, flow_threshold = STATUS_SALVAGED, salvage_flow_threshold
            except Exception as e2:
                print(f"[error] {worker} {name}: {type(e2).__name__}: {e2}", flush=True)
                if use_gpu:
                    torch.cuda.empty_cache()
                write_error_mask(out_mask, dapi.shape, name, worker, e2, salvage_flow_threshold)
                del dapi
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{consecutive_failures} cores failed back to back on {worker}; "
                        "aborting instead of writing placeholders for the rest")
                continue

        n_cells = int(masks.max())
        print(f"  {worker} eval={time.time()-t1:.1f}s n_cells={n_cells} status={status}", flush=True)

        if n_cells >= 65535:
            print(f"[warn] {worker} {name} has {n_cells} labels; writing uint32", flush=True)
        masks_out = masks.astype(np.uint16) if n_cells < 65535 else masks.astype(np.uint32)
        _write_mask(out_mask, masks_out, status, n_cells, flow_threshold)
        if status == STATUS_SALVAGED:
            _write_sidecar(sidecar_path(out_mask, "salvaged"),
                           {"core": name, "status": STATUS_SALVAGED, "n_cells": n_cells,
                            "flow_threshold": salvage_flow_threshold,
                            "note": "cellpose QC step (flow_threshold) disabled for this core",
                            "time": time.strftime("%Y-%m-%dT%H:%M:%S")})
        else:
            # A core that used to fail and now segments cleanly should not keep its sidecar.
            _clear_sidecars(out_mask)
        print(f"[done] {worker} {name} total={time.time()-t0:.1f}s", flush=True)
        consecutive_failures = 0

        # Release this core's tensors before the next one: the QC step allocates
        # ~9 GB per 100M mask pixels and fragmentation accumulates across cores.
        del masks, flows, styles, masks_out, dapi
        if use_gpu:
            torch.cuda.empty_cache()

    print(f"[all done] {worker}", flush=True)


def summarize(output_dir, files):
    """Report how every core ended up, read from the masks' own headers."""
    counts = {}
    for mask_path in sorted(glob.glob(os.path.join(output_dir, "*.tif"))):
        status, _ = read_mask_status(mask_path)
        key = status or "legacy"
        counts[key] = counts.get(key, 0) + 1
    written = sum(counts.values())
    breakdown = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    print(f"[summary] total={len(files)} masks={written} missing={len(files) - written} "
          f"{breakdown}", flush=True)
    for kind in ("error", "salvaged"):
        for sidecar in sorted(glob.glob(os.path.join(output_dir, f"*.{kind}.json"))):
            print(f"[summary] {kind}: {os.path.basename(sidecar)}", flush=True)


def main():
    args = parse_args()
    if args.n_gpu < 1:
        raise ValueError("--n_gpu must be at least 1")

    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.input_dir, "*.IF.ome.tif")))
    files += sorted(glob.glob(os.path.join(args.input_dir, "*.IF.ome.tiff")))
    print(f"[init] found {len(files)} files", flush=True)

    if args.n_gpu == 1 or not torch.cuda.is_available():
        if args.n_gpu > 1:
            print("[warn] CUDA is not available; running on CPU with one worker", flush=True)
        process_files(files, args.output_dir, gpu_index=0 if torch.cuda.is_available() else None,
                      dapi_channel=args.dapi_channel,
                      salvage_flow_threshold=args.salvage_flow_threshold)
        summarize(args.output_dir, files)
        print("[all done]", flush=True)
        return

    available_gpus = torch.cuda.device_count()
    if args.n_gpu > available_gpus:
        raise ValueError(f"--n_gpu={args.n_gpu} but only {available_gpus} CUDA device(s) are visible")

    ctx = mp.get_context("spawn")
    processes = []
    for gpu_index in range(args.n_gpu):
        gpu_files = files[gpu_index::args.n_gpu]
        process = ctx.Process(target=process_files,
                              args=(gpu_files, args.output_dir, gpu_index, args.dapi_channel,
                                    args.salvage_flow_threshold))
        process.start()
        processes.append(process)

    failed = []
    for gpu_index, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            failed.append((gpu_index, process.exitcode))

    summarize(args.output_dir, files)

    if failed:
        details = ", ".join(f"gpu {gpu}: exit {code}" for gpu, code in failed)
        raise RuntimeError(f"Cellpose worker failed: {details}")

    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
