import re
from pathlib import Path

import dask.array as da
import numpy as np
import tifffile
import zarr
from lxml import etree
from loguru import logger


def get_ome_tiff_description_xml(**metadata):
    sizeC,sizeX,sizeY=metadata["sizeC"],metadata["sizeX"],metadata["sizeY"]
    dtype_str = metadata["dtype"]
    NSMAP = {
        None: "http://www.openmicroscopy.org/Schemas/OME/2016-06",
        "xsi": "http://www.w3.org/2001/XMLSchema-instance"
    }
    
    ome = etree.Element("OME", nsmap=NSMAP,
                        UUID="urn:uuid:c3910fd4-0511-11f0-9999-08bfb88190b7",
                        Creator="tifffile.py 2025.3.13")
    
    image = etree.SubElement(ome, "Image", ID="Image:0", Name="Image0")
    etree.SubElement(image, "Description").text = metadata.get("Description",None)
    etree.SubElement(image, "InstrumentRef", ID="Instrument:0")
    etree.SubElement(image, "ObjectiveSettings", ID="Objective:0")
    
    pixels = etree.SubElement(
        image, "Pixels", ID="Pixels:0", DimensionOrder="XYCZT", Type=dtype_str,
        SizeX=str(sizeX), SizeY=str(sizeY), SizeC=str(sizeC), SizeZ="1", SizeT="1",
        PhysicalSizeX=metadata.get("PhysicalSizeX",""), PhysicalSizeXUnit=metadata.get("PhysicalSizeXUnit",""),
        PhysicalSizeY=metadata.get("PhysicalSizeY",""), PhysicalSizeYUnit=metadata.get("PhysicalSizeYUnit","")
    )
    
    for i in range(sizeC):
        if 'Channel' in metadata and 'Name' in metadata['Channel']:
            channel_name = metadata['Channel']['Name'][i]
        else:
            channel_name = f"Channel {i}"
        channel = etree.SubElement(pixels, "Channel", ID=f"Channel:0:{i}", SamplesPerPixel="1", Name=channel_name)
        etree.SubElement(channel, "LightPath")
    
    etree.SubElement(pixels, "TiffData", IFD="0", PlaneCount=str(sizeC))
    
    instrument = etree.SubElement(ome, "Instrument", ID="Instrument:0")
    etree.SubElement(instrument, "Objective", ID="Objective:0", NominalMagnification=metadata.get("NominalMagnification",""))
    
    xml_byte_string = etree.tostring(ome, pretty_print=True, encoding='UTF-8', xml_declaration=True)
    return xml_byte_string

def parse_qptiff_channel_marker_names(tiff):
    channel_marker_names = list()
    for page in tiff.series[0].levels[0].pages:
        xml_string = page.tags.get("ImageDescription").value.encode('utf-16')
        parser = etree.XMLParser(remove_blank_text=True)
        root = etree.XML(xml_string, parser)
        res = root.find("Biomarker")
        assert res is not None, "Biomarker not found in ImageDescription"
        channel_marker_names.append(res.text)
    return channel_marker_names

def parse_ome_tiff_channel_marker_names(tiff):
    import lxml.etree as etree
    root = etree.fromstring(tiff.pages[0].tags["ImageDescription"].value.encode("utf-8"))
    ns = {'ome': 'http://www.openmicroscopy.org/Schemas/OME/2016-06'}
    channels = root.xpath('//ome:Channel', namespaces=ns)
    channel_names = [ch.get("Name","") for ch in channels]
    return channel_names


def _parse_svs_pixel_size_um(tif):
    """Parse microns-per-pixel from SVS (Aperio) ImageDescription. Returns None if not found."""
    try:
        page = tif.pages[0]
        desc_tag = page.tags.get("ImageDescription")
        if desc_tag is None:
            return None
        desc = desc_tag.value
        if desc is None or not isinstance(desc, str):
            return None
        match = re.search(r"(?:aperio\.)?MPP\s*=\s*([\d.]+)", desc, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def _parse_svs_magnification(tif):
    """Parse apparent magnification from SVS (Aperio) ImageDescription. Returns None if not found."""
    try:
        page = tif.pages[0]
        desc_tag = page.tags.get("ImageDescription")
        if desc_tag is None:
            return None
        desc = desc_tag.value
        if desc is None or not isinstance(desc, str):
            return None
        match = re.search(r"(?:aperio\.)?AppMag\s*=\s*([\d.]+)", desc, re.IGNORECASE)
        if match:
            return float(match.group(1))
    except Exception:
        pass
    return None


def _get_wsi_dask_svs(tif):
    """Level-0 dask array (C, H, W) and metadata dict (pixel_size, magnification, channel_names)."""
    page = tif.pages[0]
    arr = da.from_zarr(page.aszarr())
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = arr.transpose([2, 0, 1])
    elif arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    channel_names = ["Red", "Green", "Blue"] if arr.shape[0] == 3 else None
    pixel_size = _parse_svs_pixel_size_um(tif)
    if pixel_size is None:
        pixel_size = 0.5072
        logger.warning(
            "Pixel size could not be read from SVS metadata; using default {:.4f} µm",
            pixel_size,
        )
    magnification = _parse_svs_magnification(tif)
    if magnification is None:
        magnification = 20.0
        logger.warning(
            "Magnification could not be read from SVS metadata; using default {:.4f}",
            magnification,
        )
    metadata = {
        "pixel_size": pixel_size,
        "magnification": magnification,
        "channel_names": channel_names,
    }
    return arr, metadata


def _get_wsi_dask_ometiff(tif):
    """Level-0 dask array (C, H, W) and metadata dict (pixel_size, magnification, channel_names)."""
    try:
        level = tif.series[0].levels[0]
    except (IndexError, AttributeError):
        raise ValueError("Not a valid OME-TIFF / QPTIFF (no series[0].levels[0])")

    store = level.aszarr()
    group = zarr.open(store, mode="r")

    if isinstance(group, zarr.hierarchy.Group):
        if "0" in group:
            zarr_arr = group["0"]
        else:
            keys = sorted(group.array_keys()) or list(group.keys())
            if not keys:
                raise ValueError("No array found in OME-TIFF series level")
            zarr_arr = group[keys[0]]
    else:
        zarr_arr = group

    arr = da.from_zarr(zarr_arr)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    elif arr.ndim == 3:
        if arr.shape[0] < arr.shape[1] and arr.shape[0] < arr.shape[2]:
            pass
        elif arr.shape[2] in (3, 4) and arr.shape[0] > 4:
            arr = arr.transpose([2, 0, 1])

    channel_names = None
    pixel_size = None
    magnification = None
    try:
        if hasattr(tif, "series") and tif.series and tif.series[0].levels[0].pages:
            first_page = tif.series[0].levels[0].pages[0]
            if first_page.tags.get("ImageDescription"):
                desc = first_page.tags["ImageDescription"].value
                if "Biomarker" in str(desc) or "utf-16" in str(desc):
                    channel_names = parse_qptiff_channel_marker_names(tif)
                else:
                    channel_names = parse_ome_tiff_channel_marker_names(tif)
    except Exception:
        pass

    try:
        if tif.pages and tif.pages[0].tags.get("ImageDescription"):
            root = etree.fromstring(
                tif.pages[0].tags["ImageDescription"].value.encode("utf-8")
            )
            ns = {"ome": "http://www.openmicroscopy.org/Schemas/OME/2016-06"}
            px = root.find(".//ome:Pixels", namespaces=ns)
            if px is not None:
                if px.get("PhysicalSizeX"):
                    pixel_size = float(px.get("PhysicalSizeX"))
                elif px.get("PhysicalSizeY"):
                    pixel_size = float(px.get("PhysicalSizeY"))
            obj = root.find(".//ome:Objective", namespaces=ns)
            if obj is not None and obj.get("NominalMagnification"):
                magnification = float(obj.get("NominalMagnification"))
    except Exception:
        pass

    if pixel_size is None:
        pixel_size = 0.5072
        logger.warning(
            "Pixel size could not be read from OME-TIFF metadata; using default {:.4f} µm",
            pixel_size,
        )
    if magnification is None:
        magnification = 20.0
        logger.warning(
            "Magnification could not be read from OME-TIFF metadata; using default {:.4f}",
            magnification,
        )

    metadata = {
        "pixel_size": pixel_size,
        "magnification": magnification,
        "channel_names": channel_names,
    }
    return arr, metadata


def open_wsi(wsi_path):
    """
    Open WSI and return (TiffFile handle, level-0 dask array (C,H,W), metadata).

    metadata is a dict with keys:
      - pixel_size: float (µm per pixel)
      - magnification: float (nominal magnification, e.g. 20.0)
      - channel_names: list[str] | None (per-channel names if available)

    Caller must close the TiffFile when done.
    """
    wsi_path = Path(wsi_path)
    suffix = wsi_path.suffix.lower()
    if suffix == ".svs":
        tif = tifffile.TiffFile(wsi_path)
        arr, metadata = _get_wsi_dask_svs(tif)
    elif suffix == ".qptiff" or (suffix in (".tif", ".tiff") and "ome" in wsi_path.name.lower()):
        tif = tifffile.TiffFile(wsi_path)
        arr, metadata = _get_wsi_dask_ometiff(tif)
    else:
        tif = tifffile.TiffFile(wsi_path)
        try:
            arr, metadata = _get_wsi_dask_ometiff(tif)
        except (ValueError, AttributeError):
            tif.close()
            raise ValueError(
                f"Unsupported format or structure for {wsi_path}. "
                "Use .svs for HE or .qptiff/.ome.tiff for multiplex."
            )
    return tif, arr, metadata



def write_ome_tiff(
    arr,
    output_path,
    *,
    pixel_size_um = 0.5072,
    channel_names=None,
    description="",
    magnification=20.0,
    downscale_factor=4,
    compression="zlib",
    maxworkers=15,
):
    """
    Write a multi-channel array (C, H, W) to pyramidal OME-TIFF using palom.

    Parameters
    ----------
    arr : dask.array.Array or numpy.ndarray
        Shape (C, H, W), any numeric dtype (e.g. uint8, uint16).
    output_path : str or pathlib.Path
        Path for the output OME-TIFF file.
    pixel_size_um : float
        Physical pixel size in microns (same for X and Y).
    channel_names : list of str, optional
        Names for each channel; length must match C. If None, uses Channel_0, Channel_1, ...
    description : str, optional
        Image description stored in OME metadata.
    magnification : float, optional
        Nominal magnification (default 20.0).
    downscale_factor : int, optional
        Pyramid level downscale factor (default 4).
    compression : str, optional
        TIFF compression: 'zlib', 'lzw', 'deflate', or 'none' (default 'zlib').
    maxworkers : int, optional
        Max workers for tifffile (default 15).
    """
    import palom
    C, H, W = arr.shape
    if channel_names is None:
        channel_names = [f"Channel_{i}" for i in range(C)]
    if len(channel_names) != C:
        channel_names = [f"Channel_{i}" for i in range(C)]
    metadata = {
        "Description": description,
        "PhysicalSizeX": str(pixel_size_um),
        "PhysicalSizeXUnit": "µm",
        "PhysicalSizeY": str(pixel_size_um),
        "PhysicalSizeYUnit": "µm",
        "NominalMagnification": str(magnification),
        "Channel": {"Name": channel_names},
    }
    description_xml = get_ome_tiff_description_xml(
        sizeC=C,
        sizeX=W,
        sizeY=H,
        dtype=str(arr.dtype),
        **metadata,
    )
    palom.pyramid.write_pyramid(
        mosaics=[arr],
        output_path=str(output_path),
        pixel_size=pixel_size_um,
        downscale_factor=downscale_factor,
        compression=compression,
        kwargs_tifffile=dict(
            description=description_xml,
            maxworkers=maxworkers,
            photometric="minisblack",
        ),
    )


