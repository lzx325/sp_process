"""Run cellpose-SAM on the DAPI channel of each CODEX_registered OME-TIFF.

Usage:
    python cellpose_detection.py --input-dir INPUT_DIR --output-dir OUTPUT_DIR \
        [--n_gpu N] [--dapi_channel C]

If --n_gpu is greater than 1, the input files are split across multiple
Cellpose worker processes, one per visible GPU.
"""
import os
import re
import argparse
import glob
import multiprocessing as mp
import time
import numpy as np
import tifffile
import torch
from cellpose import models

DAPI_CHANNEL = 0


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", required=True, help="Directory of CODEX_registered OME-TIFF files.")
    parser.add_argument("--output-dir", required=True, help="Directory to write cellpose masks.")
    parser.add_argument("--n_gpu", type=int, default=1, help="Number of GPUs to use (default: 1).")
    parser.add_argument("--dapi_channel", type=int, default=DAPI_CHANNEL,
                        help=f"Index of the DAPI channel in the OME-TIFF (default: {DAPI_CHANNEL}).")
    return parser.parse_args()


def process_files(files, output_dir, gpu_index=None, dapi_channel=DAPI_CHANNEL):
    use_gpu = gpu_index is not None and torch.cuda.is_available()
    device = torch.device(f"cuda:{gpu_index}") if use_gpu else torch.device("cpu")
    worker = f"gpu={gpu_index}" if use_gpu else "cpu"
    print(f"[init] {worker} device={device} files={len(files)}", flush=True)

    model = models.CellposeModel(gpu=use_gpu, device=device)

    for f in files:
        name = re.sub(r"\.IF\.ome\.tiff?$", "", os.path.basename(f))
        out_mask = os.path.join(output_dir, f"{name}.tif")
        if os.path.exists(out_mask):
            print(f"[skip] {worker} {name}", flush=True)
            continue

        t0 = time.time()
        print(f"[load] {worker} {name}", flush=True)
        try:
            with tifffile.TiffFile(f) as tif:
                dapi = tif.series[0].asarray(key=dapi_channel)
        except Exception as e:
            print(f"[error] {worker} {name}: {e}", flush=True)
            continue
        print(f"  {worker} shape={dapi.shape} dtype={dapi.dtype} load={time.time()-t0:.1f}s", flush=True)

        t1 = time.time()
        masks, flows, styles = model.eval(dapi)
        n_cells = int(masks.max())
        print(f"  {worker} eval={time.time()-t1:.1f}s n_cells={n_cells}", flush=True)

        masks_out = masks.astype(np.uint16) if n_cells < 65535 else masks.astype(np.uint32)
        tmp = out_mask + ".tmp"
        tifffile.imwrite(tmp, masks_out, compression="zlib", bigtiff=True)
        os.replace(tmp, out_mask)
        print(f"[done] {worker} {name} total={time.time()-t0:.1f}s", flush=True)

    print(f"[all done] {worker}", flush=True)


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
                      dapi_channel=args.dapi_channel)
        print("[all done]", flush=True)
        return

    available_gpus = torch.cuda.device_count()
    if args.n_gpu > available_gpus:
        raise ValueError(f"--n_gpu={args.n_gpu} but only {available_gpus} CUDA device(s) are visible")

    ctx = mp.get_context("spawn")
    processes = []
    for gpu_index in range(args.n_gpu):
        gpu_files = files[gpu_index::args.n_gpu]
        process = ctx.Process(target=process_files, args=(gpu_files, args.output_dir, gpu_index, args.dapi_channel))
        process.start()
        processes.append(process)

    failed = []
    for gpu_index, process in enumerate(processes):
        process.join()
        if process.exitcode != 0:
            failed.append((gpu_index, process.exitcode))

    if failed:
        details = ", ".join(f"gpu {gpu}: exit {code}" for gpu, code in failed)
        raise RuntimeError(f"Cellpose worker failed: {details}")

    print("[all done]", flush=True)


if __name__ == "__main__":
    main()
