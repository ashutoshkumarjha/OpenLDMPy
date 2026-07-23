"""Shapefile -> multi-band class-weight raster conversion.

Port of R's ``rasterise()``/``createMutliRaster()`` (Rasterise_dev_68akj.r
lines 1369-1444), deferred at Slice 1. Follows the geopandas/
rasterio idiom established in masking.py's ``_coverage_from_vector`` rather
than R's cell-index data.table approach.

R's ``option="fraction"`` (the default) computes true polygon/cell
intersection-area fractions via ``raster::extract(weight=TRUE)``. There is
no direct rasterio/geopandas equivalent, and this feature has zero
R-oracle fixtures to validate exactness against, so this port
uses block-average supersampling (rasterize at a finer grid, then average
down) rather than exact shapely polygon-cell intersection — cheaper, no new
dependency, and consistent with the structural-only validation
precedent for paths with no fixture to check against.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from rasterio import features
from rasterio.transform import from_origin

from .config import logger
from .errors import RasterizeError
from .raster_io import RasterLayer

_VALID_OPTIONS = ("fraction", "presence")


def _class_codes(gdf, class_field: str) -> List:
    codes = list(gdf[class_field].dropna().unique())
    try:
        return sorted(codes, key=lambda c: float(c))
    except (TypeError, ValueError):
        return sorted(codes, key=str)


def _coarse_window(geom, minx: float, maxy: float, cell_size: float, height: int, width: int):
    """(row0, row1, col0, col1) coarse-grid cell range this geometry's bbox
    touches, clipped to the grid. row1/col1 are exclusive."""
    gminx, gminy, gmaxx, gmaxy = geom.bounds
    col0 = max(0, int(np.floor((gminx - minx) / cell_size)))
    col1 = min(width, int(np.ceil((gmaxx - minx) / cell_size)))
    row0 = max(0, int(np.floor((maxy - gmaxy) / cell_size)))
    row1 = min(height, int(np.ceil((maxy - gminy) / cell_size)))
    return row0, row1, col0, col1


def _rasterize_presence(gdf, class_field: str, class_codes: List, height: int, width: int, transform) -> np.ndarray:
    """Exact, binary per-class presence — one rasterize() call per class.
    Simplification vs. R: touched-cell membership, not an overlap *count*
    (R's own aggregation can in principle sum weight=1 entries from
    multiple same-class polygons touching one cell; this port treats
    presence as a plain indicator, which is what the option name implies
    and what every real caller wants)."""
    bands = np.zeros((len(class_codes), height, width), dtype="uint8")
    for i, code in enumerate(class_codes):
        shapes = [(geom, 1) for geom in gdf.loc[gdf[class_field] == code, "geometry"] if geom is not None]
        if not shapes:
            continue
        bands[i] = features.rasterize(
            shapes, out_shape=(height, width), transform=transform, fill=0, dtype="uint8", all_touched=False,
        )
    return bands


def _rasterize_fraction(
    gdf, class_field: str, class_codes: List, height: int, width: int, transform,
    supersample: int, poly_id_field: Optional[str],
) -> np.ndarray:
    """Per-polygon windowed supersample-and-average, accumulated per class
    so overlapping same-class polygons sum weight above 1 (matching R's
    actual, uncorrected behavior — R's errorpoly diagnostic flags this
    rather than fixing it, so this port does the same)."""
    minx, maxy = transform.c, transform.f
    cell_size = transform.a
    fine_cell = cell_size / supersample
    bands = np.zeros((len(class_codes), height, width), dtype="float32")

    have_poly_id = poly_id_field is not None and poly_id_field in gdf.columns
    if poly_id_field is not None and not have_poly_id:
        logger.warning(f"poly_id_field {poly_id_field!r} not found in shapefile columns; overlap diagnostic skipped.")

    for _, row in gdf.iterrows():
        geom = row.geometry
        if geom is None or geom.is_empty:
            continue
        band_i = class_codes.index(row[class_field])
        row0, row1, col0, col1 = _coarse_window(geom, minx, maxy, cell_size, height, width)
        if row1 <= row0 or col1 <= col0:
            continue

        win_transform = from_origin(minx + col0 * cell_size, maxy - row0 * cell_size, fine_cell, fine_cell)
        fine = features.rasterize(
            [(geom, 1)],
            out_shape=((row1 - row0) * supersample, (col1 - col0) * supersample),
            transform=win_transform, fill=0, dtype="uint8", all_touched=False,
        )
        weight = fine.reshape(row1 - row0, supersample, col1 - col0, supersample).mean(axis=(1, 3))
        bands[band_i, row0:row1, col0:col1] += weight

    if have_poly_id:
        overlapping_cells = int((bands.sum(axis=0) > 1.0 + 1e-6).sum())
        if overlapping_cells:
            logger.warning(
                f"{overlapping_cells} cell(s) had summed class weight > 1 from overlapping "
                f"same-class polygons (see {poly_id_field!r} for the source polygons)."
            )
    return bands


def rasterise(
    shp_file: str,
    class_field: str = "IGBP_CODE",
    grid_size: float = 1000.0,
    option: str = "fraction",
    poly_id_field: Optional[str] = None,
    supersample: int = 10,
) -> RasterLayer:
    """Rasterize a class-coded polygon shapefile onto a fresh grid built
    from its own extent/CRS, one output band per distinct class code.

    ``option="presence"``: binary indicator per class (touched vs. not).
    ``option="fraction"`` (default): each cell's value is the fraction of
    its area covered by that class (via supersampling; ``supersample``
    trades accuracy for speed — R's own docstring flags this weighting
    step as the slow part of the original implementation too).
    """
    import geopandas as gpd

    if option not in _VALID_OPTIONS:
        raise RasterizeError(f"option must be one of {_VALID_OPTIONS}, got {option!r}")

    gdf = gpd.read_file(shp_file)
    if class_field not in gdf.columns:
        raise RasterizeError(
            f"class_field {class_field!r} not found in {shp_file}. Available columns: {list(gdf.columns)}"
        )
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty]
    if len(gdf) == 0:
        raise RasterizeError(f"{shp_file} has no usable (non-null, non-empty) geometries.")

    class_codes = _class_codes(gdf, class_field)
    minx, miny, maxx, maxy = gdf.total_bounds
    width = max(1, int(np.ceil((maxx - minx) / grid_size)))
    height = max(1, int(np.ceil((maxy - miny) / grid_size)))
    transform = from_origin(minx, maxy, grid_size, grid_size)

    logger.info(f"Rasterizing {shp_file}: {len(class_codes)} classes, {height}x{width} grid @ {grid_size}")
    if option == "presence":
        bands = _rasterize_presence(gdf, class_field, class_codes, height, width, transform)
        dtype, nodata = "uint8", 0
    else:
        bands = _rasterize_fraction(gdf, class_field, class_codes, height, width, transform, supersample, poly_id_field)
        dtype, nodata = "float32", 0.0

    profile = {
        "driver": "GTiff",
        "height": height,
        "width": width,
        "count": len(class_codes),
        "crs": gdf.crs,
        "transform": transform,
        "dtype": dtype,
        "nodata": nodata,
    }
    return RasterLayer(array=bands.astype(dtype), profile=profile, band_names=[str(c) for c in class_codes])
