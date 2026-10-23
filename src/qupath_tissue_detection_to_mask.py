import argparse
import json
import os
import sys
import geopandas as gpd
import numpy as np
import pandas as pd
import tifffile
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from geometry_tools import polygon_to_mask

IMAGE_EXTENSIONS = {'.tif', '.tiff', '.svs', '.png', '.jpg', '.jpeg', '.ndpi', '.scn'}


def get_image_dimensions(image_path):
    """Return (width, height) of the full-resolution image without loading pixels."""
    if os.path.splitext(image_path)[1].lower() in {'.tif', '.tiff', '.svs', '.ndpi', '.scn'}:
        with tifffile.TiffFile(image_path) as tif:
            series = tif.series[0]
            axes, shape = series.axes, series.shape
            return shape[axes.index('X')], shape[axes.index('Y')]
    # Whole-slide images exceed PIL's decompression-bomb limit; we only read the header.
    Image.MAX_IMAGE_PIXELS = None
    with Image.open(image_path) as img:
        width, height = img.size
    return width, height


def get_file_prefix(filename):
    """Strip all extensions from filename to get the base prefix."""
    name = filename
    while True:
        base, ext = os.path.splitext(name)
        if not ext:
            break
        name = base
    return name


def build_image_index(image_dir):
    """Map file prefix -> full image path."""
    index = {}
    for fname in os.listdir(image_dir):
        _, ext = os.path.splitext(fname)
        if ext.lower() in IMAGE_EXTENSIONS:
            prefix = get_file_prefix(fname)
            index[prefix] = os.path.join(image_dir, fname)
    return index


def main():
    parser = argparse.ArgumentParser(
        description="Convert QuPath tissue detection GeoJSON exports to binary mask images."
    )
    parser.add_argument("--image_dir", required=True,
                        help="Directory containing the source images.")
    parser.add_argument("--geojson_dir", required=True,
                        help="Directory containing the QuPath-exported GeoJSON files.")
    parser.add_argument("--output_dir", required=True,
                        help="Directory to save the output mask images.")
    parser.add_argument("--classification_col", default=None,
                        help="Column name in the GeoJSON properties to use for mask values (optional).")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    image_index = build_image_index(args.image_dir)

    geojson_files = [f for f in os.listdir(args.geojson_dir) if f.lower().endswith('.geojson')]
    if not geojson_files:
        sys.exit(f"No GeoJSON files found in {args.geojson_dir}")

    fatal = []    # problems that must stop the pipeline
    blank = []    # cores that got an all-zero mask instead of real tissue
    written = set()

    for geojson_fname in tqdm(sorted(geojson_files), desc="Converting GeoJSON to mask"):
        geojson_path = os.path.join(args.geojson_dir, geojson_fname)
        prefix = get_file_prefix(geojson_fname)

        if prefix not in image_index:
            print(f"  [ERROR] No matching image found for '{geojson_fname}' (prefix: '{prefix}')")
            fatal.append(f"{geojson_fname}: no matching image (prefix '{prefix}')")
            continue

        image_path = image_index[prefix]
        print(f"Processing: {geojson_fname} -> {os.path.basename(image_path)}")

        try:
            image_width, image_height = get_image_dimensions(image_path)
        except Exception as e:
            print(f"  [ERROR] Could not read image dimensions from '{image_path}': {e}")
            fatal.append(f"{geojson_fname}: unreadable image '{image_path}': {e}")
            continue

        try:
            with open(geojson_path, "r") as f:
                geojson_data = json.load(f)
            # QuPath may export a bare list of features or a FeatureCollection
            if isinstance(geojson_data, list):
                features = geojson_data
            else:
                features = geojson_data.get("features", [])
            gdf = gpd.GeoDataFrame.from_features(features)
        except Exception as e:
            print(f"  [ERROR] Could not read GeoJSON '{geojson_path}': {e}")
            fatal.append(f"{geojson_fname}: unreadable GeoJSON: {e}")
            continue

        output_path = os.path.join(args.output_dir, prefix + "_mask.tif")

        # An all-zero mask is the right answer for a core QuPath found no tissue in, and
        # the fallback when polygons cannot be rasterised. Either way the core keeps a
        # mask file: downstream then sees a core with no tissue instead of silently
        # matching a neighbour's mask by name prefix.
        mask = None
        if gdf.empty:
            print(f"  [BLANK] GeoJSON has no features: {geojson_fname}")
            blank.append(f"{prefix}: GeoJSON has no features")
        else:
            metadata_df = pd.DataFrame(gdf.drop(columns='geometry')).reset_index(drop=True)
            try:
                mask = polygon_to_mask(
                    shapes_list=list(gdf.geometry),
                    metadata_df=metadata_df,
                    classification_col=args.classification_col,
                    image_width=image_width,
                    image_height=image_height,
                )
            except Exception as e:
                print(f"  [ERROR] Failed to create mask for '{geojson_fname}': {e}")
                blank.append(f"{prefix}: {type(e).__name__}: {e}")

        if mask is None:
            mask = np.zeros((image_height, image_width), dtype=np.uint32)

        # Write via a pid-unique tmp name so a killed job leaves no half-written TIFF
        # for step 6 to read as a valid mask.
        try:
            tmp = f"{output_path}.{os.getpid()}.tmp"
            tifffile.imwrite(tmp, mask.astype("uint32"), compression='zlib')
            os.replace(tmp, output_path)
        except Exception as e:
            print(f"  [ERROR] Could not write '{output_path}': {e}")
            fatal.append(f"{prefix}: could not write mask: {e}")
            continue

        written.add(prefix)
        print(f"  Saved mask: {output_path}")

    # Every image must end up with a mask: a gap is what lets step 6 pick up the wrong
    # one, so it fails the step rather than being left for a reader to notice.
    missing = sorted(set(image_index) - written)
    if missing:
        fatal.append(f"{len(missing)} image(s) have no mask: {missing[:10]}")

    print(f"[summary] images={len(image_index)} masks={len(written)} "
          f"blank={len(blank)} failed={len(fatal)}")
    for note in blank:
        print(f"[summary] blank-mask {note}")

    if fatal:
        for note in fatal:
            print(f"[fatal] {note}")
        sys.exit(f"Tissue mask conversion failed for {len(fatal)} item(s); see [fatal] above")

if __name__ == "__main__":
    main()
