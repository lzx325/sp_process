import os
import sys
from pathlib import Path
import shutil
import zarr
import dask.array as da
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tifffile
from IPython.display import display
import anndata as ad
import pandas as pd
import scanpy as sc
from tqdm import tqdm
import argparse
from loguru import logger

from sp_marker_definitions import CELL_TYPE_MARKERS_TMA2025

sc.settings.verbosity = 3

def subset_markers(adata, marker_dict):
    markers = list(dict.fromkeys(m for ms in marker_dict.values() for m in ms))
    markers_present = [m for m in markers if m in adata.var_names]
    missing = set(markers) - set(markers_present)
    if missing:
        logger.warning(f"Markers not found in adata and will be skipped: {missing}")
    return adata[:, markers_present].copy()

def preprocess(adata, use_bbknn=False):
    adata.X[np.isnan(adata.X)] = 0
    adata.layers["raw"] = adata.X.copy()
    sc.pp.scale(adata)
    sc.tl.pca(adata, svd_solver='arpack')
    if use_bbknn:
        sc.external.pp.bbknn(adata, batch_key="sample",trim=30)
    else:
        sc.pp.neighbors(adata)

def leiden(adata):
    sc.tl.leiden(adata)

def umap(adata):
    sc.tl.umap(adata)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Cluster cell types based on marker expressions")
    parser.add_argument("--input_anndata_fp", required=True, help="File path for the input AnnData (.h5ad) containing cell marker expressions")
    parser.add_argument("--output_dir", required=True, help="Directory to save the output files")
    parser.add_argument("--run_preprocess", action="store_true", help="Whether to run the preprocess step")
    parser.add_argument("--run_leiden", action="store_true", help="Whether to run the Leiden clustering step")
    parser.add_argument("--run_umap", action="store_true", help="Whether to run the UMAP dimensionality reduction step")
    parser.add_argument("--use_bbknn", action="store_true", help="Build the neighbor graph with BBKNN (batch_key='sample') instead of sc.pp.neighbors")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fp_preprocessed = output_dir / "01_preprocessed.h5ad"
    fp_leiden       = output_dir / "02_leiden.h5ad"
    fp_umap         = output_dir / "03_umap.h5ad"

    # step 1: loading and preprocessing
    if args.run_preprocess or not fp_preprocessed.exists():
        logger.info("Running step 1: preprocessing")
        adata_raw = ad.read_h5ad(args.input_anndata_fp)
        logger.info(f"Loaded AnnData with {adata_raw.n_obs} cells and {adata_raw.n_vars} markers")
        # adata = subset_markers(adata_raw, CELL_TYPE_MARKERS_TMA2025) 
        adata = subset_markers(adata_raw, CELL_TYPE_MARKERS_TMA2025)
        preprocess(adata, use_bbknn=args.use_bbknn)
        adata.write_h5ad(fp_preprocessed)
        logger.info(f"Saved preprocessed anndata to {fp_preprocessed}")
    else:
        logger.info(f"Skipping step 1: {fp_preprocessed} already exists")

    # step 2: Leiden clustering
    if args.run_leiden or not fp_leiden.exists():
        logger.info("Running step 2: Leiden clustering")
        adata = ad.read_h5ad(fp_preprocessed)
        logger.info(f"Running Leiden clustering on {adata.n_obs} cells and {adata.n_vars} markers")
        leiden(adata)
        adata.write_h5ad(fp_leiden)
        logger.info(f"Saved Leiden anndata to {fp_leiden}")
    else:
        logger.info(f"Skipping step 2: {fp_leiden} already exists")

    # step 3: UMAP
    if args.run_umap or not fp_umap.exists():
        logger.info("Running step 3: UMAP")
        adata = ad.read_h5ad(fp_leiden)
        logger.info(f"Running UMAP on {adata.n_obs} cells and {adata.n_vars} markers")
        umap(adata)
        adata.write_h5ad(fp_umap)
        logger.info(f"Saved UMAP anndata to {fp_umap}")
    else:
        logger.info(f"Skipping step 3: {fp_umap} already exists")