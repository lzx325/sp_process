import argparse
import math
import os
import sys
from pathlib import Path
from glob import glob

import cv2 as cv
import dask.array
import matplotlib.pyplot as plt
import numpy as np
import tifffile
import zarr
from lxml import etree
from loguru import logger

import palom
from tif_utils import open_wsi, write_ome_tiff

cv.setNumThreads(30)

def _image_resize(image: np.ndarray, ds: int) -> np.ndarray:
    assert len(image.shape) in [2,3]
    if ds == 1:
        return image.copy()
    if len(image.shape) == 2:
        image_resized = cv.resize(image, None, fx = 1/ds, fy = 1/ds, interpolation=cv.INTER_AREA)
    elif len(image.shape) == 3:
        if image.shape[0] == 3:
            image = image.transpose([1,2,0])
        else:
            assert image.shape[2] == 3
        image_resized = cv.resize(image, None, fx = 1/ds, fy = 1/ds, interpolation=cv.INTER_AREA)
    return image_resized

def _read_WSI(wsi_fp):
    import palom
    
    tif, _arr, metadata = open_wsi(wsi_fp)
    try:
        if str(wsi_fp).endswith('.svs'):
            palom_reader = palom.reader.SvsReader(wsi_fp)
        elif str(wsi_fp).endswith('.ome.tiff') or str(wsi_fp).endswith('.ome.tif'):
            palom_reader = palom.reader.OmePyramidReader(wsi_fp)
        elif str(wsi_fp).endswith('.qptiff'):
            palom_reader = palom.reader.OmePyramidReader(wsi_fp)
        else:
            raise ValueError("Unsupported file format: {}".format(wsi_fp))
        return palom_reader, metadata
    finally:
        tif.close()

def _read_HE_image(he_fp, he_downscale_factor=8, he_level = 0):
    if str(he_fp).endswith('.ome.tiff') or str(he_fp).endswith('.ome.tif') or str(he_fp).endswith('.qptiff') or str(he_fp).endswith('.svs'):
        he_palom_reader, he_metadata = _read_WSI(he_fp)
        he_palom_processor = palom.color.PyramidHaxProcessor(he_palom_reader.pyramid, thumbnail_level=he_level)
        he_gs = he_palom_processor.get_processed_color(0,mode='grayscale')
        he_gs_values = he_gs.compute()
        he_gs_thumbnail = _image_resize(he_gs_values,he_downscale_factor)
    elif str(he_fp).endswith('.png') or str(he_fp).endswith('.jpg') or str(he_fp).endswith('.jpeg'):
        logger.info("HE image is a PNG, JPEG, or JPG file, he_level ignored")
        he_gs_values = cv.imread(str(he_fp), cv.IMREAD_GRAYSCALE)
        he_gs = dask.array.from_array(he_gs_values, chunks=(512, 512))
        he_gs_thumbnail = _image_resize(he_gs_values, he_downscale_factor)
        he_metadata = None
    else:
        raise ValueError(f"Unsupported HE file format: {ext}")
    return he_gs, he_gs_thumbnail, he_metadata

def _align_image(
    ref_img, moving_img, ref_thumbnail, moving_thumbnail,
    output_dir=None,
    n_keypoints=10000,
):
    """
    Run block affine alignment of moving image to reference image.

    Returns
    -------
    block_affine_matrices_da : dask.array.Array
        Per-block affine matrices for transforming the moving image.
    """
    ref_thumbnail_down_factor = ref_img.shape[0] / ref_thumbnail.shape[0]
    moving_thumbnail_down_factor = moving_img.shape[0] / moving_thumbnail.shape[0]

    logger.info(f"ref_thumbnail_down_factor: {ref_thumbnail_down_factor}")
    logger.info(f"moving_thumbnail_down_factor: {moving_thumbnail_down_factor}")

    aligner = palom.align.Aligner(
        ref_img=ref_img,
        moving_img=moving_img,
        ref_thumbnail=ref_thumbnail,
        moving_thumbnail=moving_thumbnail,
        ref_thumbnail_down_factor=ref_thumbnail_down_factor,
        moving_thumbnail_down_factor=moving_thumbnail_down_factor,
    )

    aligner.coarse_register_affine(n_keypoints=n_keypoints)

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        coarse_affine_matrix_fp = os.path.join(output_dir, "coarse_affine_matrix.npy")
        np.save(coarse_affine_matrix_fp, aligner.coarse_affine_matrix)
        logger.info(f"Coarse affine matrix saved to {coarse_affine_matrix_fp}")

        full_res_coarse_affine_matrix_fp = os.path.join(
            output_dir, "full_res_coarse_affine_matrix.npy"
        )
        np.save(full_res_coarse_affine_matrix_fp, aligner.affine_matrix)
        logger.info(
            f"Full-resolution coarse affine matrix saved to {full_res_coarse_affine_matrix_fp}"
        )

        alignment_image_fp = os.path.join(output_dir, "alignment_image.png")
        plt.gcf().savefig(alignment_image_fp, dpi=300)
        logger.info(f"Alignment image saved to {alignment_image_fp}")

    aligner.compute_shifts()
    aligner.constrain_shifts()

    block_affine_matrices_da = aligner.block_affine_matrices_da.persist()

    if output_dir is not None:

        fine_affine_matrices_fp = os.path.join(
            output_dir, "fine_grained_affine_matrices.zarr"
        )
        block_affine_matrices_da.to_zarr(fine_affine_matrices_fp, overwrite=True)
        logger.info(
            f"Fine-grained affine matrices saved to {fine_affine_matrices_fp}"
        )

    return block_affine_matrices_da

def register_codex_to_HE(
    codex_fp,
    he_fp,
    output_dir,
    *,
    he_level=0,
    he_downscale_factor=8,
    codex_level=0,
    codex_downscale_factor=8,
    codex_reg_channel=0,
    pixel_size=None,
    magnification=None,
    align_he_to_codex=False,
):
    def _get_codex_downscale_factor(codex_downscale_factor, he_H, he_W, codex_H, codex_W):
        size_multiples = math.sqrt(he_H * he_W) / math.sqrt(codex_H * codex_W)
        rounded_size_multiples = int(2 ** int(round(math.log2(size_multiples))))
        return codex_downscale_factor / rounded_size_multiples

    curr_name = Path(codex_fp).stem.split(".")[0]
    logger.info(f"Registering {curr_name}")
    aligned_dir = os.path.join(output_dir, "aligned")
    out_fp = os.path.join(aligned_dir, curr_name + ".IF.ome.tiff")

    curr_dir = os.path.join(output_dir, curr_name)
    os.makedirs(curr_dir, exist_ok=True)
    coarse_affine_matrix_fp = os.path.join(curr_dir, "coarse_affine_matrix.npy")
    fine_affine_matrices_fp = os.path.join(
        curr_dir, "fine_grained_affine_matrices.zarr"
    )
    registration_artifacts_exist = (
        os.path.exists(coarse_affine_matrix_fp)
        and os.path.exists(fine_affine_matrices_fp)
    )
    he_to_codex_dir = os.path.join(curr_dir, "he_to_codex")
    he_to_codex_artifacts_exist = (
        os.path.exists(os.path.join(he_to_codex_dir, "coarse_affine_matrix.npy"))
        and os.path.exists(
            os.path.join(he_to_codex_dir, "fine_grained_affine_matrices.zarr")
        )
    )

    if (
        os.path.exists(out_fp)
        and registration_artifacts_exist
        and (not align_he_to_codex or he_to_codex_artifacts_exist)
    ):
        logger.info(f"Output already exists at {out_fp}, skipping registration")
        return
    if os.path.exists(out_fp) and not registration_artifacts_exist:
        logger.info(
            f"Output already exists at {out_fp}, but registration matrix artifacts are missing"
        )
    if (
        align_he_to_codex
        and os.path.exists(out_fp)
        and not he_to_codex_artifacts_exist
    ):
        logger.info(
            f"Output already exists at {out_fp}, but HE-to-CODEX matrix artifacts are missing"
        )

    codex_palom_reader, codex_metadata = _read_WSI(codex_fp)

    he_gs, he_gs_thumbnail, he_metadata = _read_HE_image(
        he_fp, he_downscale_factor, he_level
    )

    # Resolve pixel_size: command line or metadata
    if pixel_size is not None:
        level0_pixel_size = pixel_size
        logger.info(
            f"Pixel size: {level0_pixel_size} µm (specified from command line)"
        )
    else:
        if he_metadata is not None:
            level0_pixel_size = he_metadata["pixel_size"]
            logger.info(
                f"Pixel size: {level0_pixel_size} µm (from HE metadata)"
            )
        else:
            level0_pixel_size = 0.5072
            logger.info(
                f"Pixel size: {level0_pixel_size} µm (default)"
            )

    # Resolve magnification: command line or metadata
    if magnification is not None:
        nominal_magnification = magnification
        logger.info(
            f"Magnification: {nominal_magnification} (specified from command line)"
        )
    else:
        if he_metadata is not None:
            nominal_magnification = he_metadata["magnification"]
            logger.info(
                f"Magnification: {nominal_magnification} (from HE metadata)"
            )
        else:
            nominal_magnification = 20.0
            logger.info(
                f"Magnification: {nominal_magnification} (default)"
            )

    codex_img = None
    codex_thumbnail = None
    needs_alignment_run = not registration_artifacts_exist or (
        align_he_to_codex and not he_to_codex_artifacts_exist
    )
    if needs_alignment_run:
        he_H, he_W = he_gs.shape
        _, codex_H, codex_W = codex_palom_reader.pyramid[0].shape

        codex_downscale_factor = _get_codex_downscale_factor(
            codex_downscale_factor, he_H, he_W, codex_H, codex_W
        )
        assert codex_downscale_factor >= 1, "codex_downscale_factor should be >= 1"
        logger.info(
            f"he_downscale_factor: {he_downscale_factor}, codex_downscale_factor: {codex_downscale_factor}"
        )

        logger.info("Reading CODEX image: {}".format(codex_fp))
        codex_img = codex_palom_reader.read_level_channels(codex_level, codex_reg_channel)
        codex_thumbnail = _image_resize(codex_img.compute(), codex_downscale_factor)

    if not registration_artifacts_exist:
        logger.info(f"ref_img shape: {he_gs.shape}")
        logger.info(f"moving_img shape: {codex_img.shape}")
        logger.info(f"ref_thumbnail shape: {he_gs_thumbnail.shape}")
        logger.info(f"moving_thumbnail shape: {codex_thumbnail.shape}")

        block_affine_matrices_da = _align_image(
            ref_img=he_gs,
            moving_img=codex_img,
            ref_thumbnail=he_gs_thumbnail,
            moving_thumbnail=codex_thumbnail,
            output_dir=curr_dir,
        )
    else:
        logger.info(
            f"Fine-grained affine matrices already exist at {fine_affine_matrices_fp}, skipping registration"
        )
        za = zarr.open(fine_affine_matrices_fp)
        block_affine_matrices_da = dask.array.from_zarr(za)

    if align_he_to_codex:
        if not he_to_codex_artifacts_exist:
            logger.info(
                f"Registering HE to CODEX and saving matrices to {he_to_codex_dir}"
            )
            _align_image(
                ref_img=codex_img,
                moving_img=he_gs,
                ref_thumbnail=codex_thumbnail,
                moving_thumbnail=he_gs_thumbnail,
                output_dir=he_to_codex_dir,
            )
        else:
            logger.info(
                f"HE-to-CODEX matrices already exist at {he_to_codex_dir}, skipping registration"
            )

    if os.path.exists(out_fp):
        logger.info(f"Output already exists at {out_fp}, skipping image transform")
        return

    logger.info("Aligning images")
    c2m = palom.align.block_affine_transformed_moving_img(
        ref_img=he_gs,
        # select all the three channels (RGB) in moving image to transform
        moving_img=codex_palom_reader.pyramid[codex_level],
        mxs=block_affine_matrices_da
    )

    transformed_data = c2m[:,:,:]

    # Write the registered image to pyramidal OME-TIFF
    os.makedirs(aligned_dir, exist_ok=True)

    description = "Aligned CODEX image of {}".format(codex_fp)

    channel_marker_names = codex_metadata["channel_names"]
    if channel_marker_names is None:
        channel_marker_names = [f"Channel_{i}" for i in range(transformed_data.shape[0])]
        logger.info(
            "No channel names found in image metadata, using default channel names"
        )
    if len(channel_marker_names) != transformed_data.shape[0]:
        channel_marker_names = [f"Channel_{i}" for i in range(transformed_data.shape[0])]
        logger.info(
            "Number of channel names does not match number of channels, using default channel names"
        )

    write_ome_tiff(
        transformed_data,
        out_fp,
        pixel_size_um=level0_pixel_size,
        channel_names=channel_marker_names,
        magnification=nominal_magnification,
        description=description,
        downscale_factor=4,
        compression="zlib",
        maxworkers=15,
    )
    logger.info(f"Aligned image written to {out_fp}")


def main():
    parser = argparse.ArgumentParser(
        description="Register CODEX/multiplex image to HE reference and write aligned OME-TIFF."
    )
    parser.add_argument(
        "codex_fp",
        type=Path,
        help="Path to the CODEX or multiplex image (.qptiff, .ome.tiff, .ome.tif).",
    )
    parser.add_argument(
        "he_fp",
        type=Path,
        help="Path to the HE reference image (.svs, .ome.tiff, .qptiff, or .png/.jpg).",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="Output directory for alignment artifacts and aligned image.",
    )
    parser.add_argument(
        "--pixel-size",
        type=float,
        default=None,
        help="Pixel size in µm. If not set, use value from image metadata.",
    )
    parser.add_argument(
        "--magnification",
        type=float,
        default=None,
        help="Nominal magnification. If not set, use value from image metadata.",
    )
    parser.add_argument(
        "--he-level",
        type=int,
        default=0,
        help="HE pyramid level for registration (default: 0).",
    )
    parser.add_argument(
        "--he-downscale-factor",
        type=float,
        default=8,
        help="HE downscale factor for thumbnail (default: 8).",
    )
    parser.add_argument(
        "--codex-level",
        type=int,
        default=0,
        help="CODEX pyramid level for registration (default: 0).",
    )
    parser.add_argument(
        "--codex-downscale-factor",
        type=float,
        default=8,
        help="CODEX downscale factor for thumbnail (default: 8).",
    )
    parser.add_argument(
        "--codex-reg-channel",
        type=int,
        default=0,
        help="CODEX channel index used for registration (default: 0).",
    )
    parser.add_argument(
        "--align-he-to-codex",
        action="store_true",
        help=(
            "Also register HE as moving image to CODEX as reference and save only "
            "the matrix artifacts under the sample output directory."
        ),
    )
    args = parser.parse_args()

    register_codex_to_HE(
        args.codex_fp,
        args.he_fp,
        args.output_dir,
        he_level=args.he_level,
        he_downscale_factor=args.he_downscale_factor,
        codex_level=args.codex_level,
        codex_downscale_factor=args.codex_downscale_factor,
        codex_reg_channel=args.codex_reg_channel,
        pixel_size=args.pixel_size,
        magnification=args.magnification,
        align_he_to_codex=args.align_he_to_codex,
    )


if __name__ == "__main__":
    main()
