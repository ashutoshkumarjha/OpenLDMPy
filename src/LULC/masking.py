"""AOI clipping and mask exclusion for input rasters.

Port of R's ``getMasked`` (Rasterise_dev_68akj.r lines 489-557), which the
pre-Slice-1 migration accepted parameters for but never implemented.
Array-native.

Semantics preserved from R:

* ``aoi``  — keep only cells covered by the AOI (area of interest); all
  other cells become nodata.
* ``mask`` — exclude cells covered by the mask; those cells become nodata.

Both accept either a raster file (any grid — it is aligned onto the
reference grid first) or a polygon shapefile (rasterized onto the reference
grid via geopandas/rasterio; replaces R's retired ``maptools::readShapePoly``
path).
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import rasterio as rio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT

from .config import logger
from .raster_io import RasterLayer


def _is_vector(path: str) -> bool:
    return str(path).lower().endswith((".shp", ".gpkg", ".geojson", ".json"))


def _coverage_from_raster(path: str, reference: RasterLayer) -> np.ndarray:
    """Boolean (rows, cols) array on the reference grid: True where the
    mask/AOI raster has valid (non-nodata) data."""
    ref_profile = reference.profile
    with rio.open(path) as src:
        with WarpedVRT(
            src,
            crs=ref_profile.get("crs") or src.crs,
            transform=ref_profile["transform"],
            width=ref_profile["width"],
            height=ref_profile["height"],
            resampling=Resampling.nearest,
        ) as vrt:
            data = vrt.read(1)
            nodata = src.nodata
    if nodata is None:
        return np.isfinite(data)
    return data != nodata


def _coverage_from_vector(path: str, reference: RasterLayer) -> np.ndarray:
    """Boolean (rows, cols) array on the reference grid: True inside the
    shapefile's polygons."""
    import geopandas as gpd

    gdf = gpd.read_file(path)
    ref_crs = reference.profile.get("crs")
    if ref_crs is not None and gdf.crs is not None and gdf.crs != ref_crs:
        gdf = gdf.to_crs(ref_crs)
    shapes = [(geom, 1) for geom in gdf.geometry if geom is not None]
    if not shapes:
        raise ValueError(f"No usable geometries found in {path}")
    burned = features.rasterize(
        shapes,
        out_shape=reference.shape,
        transform=reference.profile["transform"],
        fill=0,
        dtype="uint8",
    )
    return burned.astype(bool)


def _coverage(path: str, reference: RasterLayer) -> np.ndarray:
    if _is_vector(path):
        return _coverage_from_vector(path, reference)
    return _coverage_from_raster(path, reference)


def apply_mask_and_aoi(
    layer: RasterLayer,
    mask_file: Optional[str] = None,
    aoi_file: Optional[str] = None,
) -> RasterLayer:
    """Return a copy of ``layer`` with AOI-outside and mask-covered cells
    set to the layer's nodata value.

    Mirrors R's ``getMasked``: AOI keeps covered cells, mask removes covered
    cells. No-op when both are None.
    """
    if mask_file is None and aoi_file is None:
        return layer

    nodata = layer.nodata
    if nodata is None:
        raise ValueError(
            "Cannot apply mask/AOI to a layer without a nodata value; "
            "pass na_value when reading the raster."
        )

    keep = np.ones(layer.shape, dtype=bool)
    if aoi_file is not None:
        logger.info(f"Clipping to AOI: {aoi_file}")
        keep &= _coverage(aoi_file, layer)
    if mask_file is not None:
        logger.info(f"Excluding mask area: {mask_file}")
        keep &= ~_coverage(mask_file, layer)

    array = layer.array.copy()
    array[:, ~keep] = nodata
    return RasterLayer(array=array, profile=layer.profile.copy(), band_names=list(layer.band_names))
