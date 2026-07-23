"""Array-native raster I/O.

Rasters are represented as :class:`RasterLayer` — a NumPy array
plus its rasterio profile — rather than being flattened into an id-keyed
DataFrame on read. This is the direct replacement for the pre-Slice-1
``data_io.DataManager`` (id/DataFrame-based) and ports the relevant parts of
R's ``getRaster``/``getDataTable``/``singleToMultiBand``/``getNumberOfClass``/
``getnoOfCell``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import rasterio as rio

from .config import NA_VALUE, logger
from .errors import DatasetValidationError


@dataclass
class RasterLayer:
    """A raster held in memory as an array plus its georeferencing profile.

    ``array`` is always 3D, shape ``(bands, rows, cols)``, even for
    single-band rasters — this keeps downstream code uniform. Use
    ``.band(0)`` or ``.single_band`` for the common single-band case.
    """

    array: np.ndarray
    profile: dict
    band_names: List[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.array.ndim == 2:
            self.array = self.array[np.newaxis, ...]
        if not self.band_names:
            self.band_names = [f"band_{i + 1}" for i in range(self.array.shape[0])]

    @property
    def shape(self) -> tuple:
        """(rows, cols) of the grid."""
        return self.array.shape[-2:]

    @property
    def count(self) -> int:
        return self.array.shape[0]

    @property
    def nodata(self):
        return self.profile.get("nodata")

    @property
    def single_band(self) -> np.ndarray:
        """The 2D array for a single-band layer. Raises if multi-band."""
        if self.array.shape[0] != 1:
            raise ValueError(f"single_band requested on a {self.array.shape[0]}-band layer")
        return self.array[0]

    def band(self, index: int) -> np.ndarray:
        return self.array[index]

    def band_by_name(self, name: str) -> np.ndarray:
        return self.array[self.band_names.index(name)]

    def valid_mask(self) -> np.ndarray:
        """Boolean (rows, cols) mask of cells that are not nodata in ANY band."""
        nodata = self.nodata
        if nodata is None:
            return np.ones(self.shape, dtype=bool)
        return np.all(self.array != nodata, axis=0)


def _open_profile(src: rio.DatasetReader, nodata_override: Optional[float] = None) -> dict:
    profile = src.profile.copy()
    if nodata_override is not None:
        profile["nodata"] = nodata_override
    return profile


def read_raster(file_path: str, na_value: Optional[float] = None) -> RasterLayer:
    """Read every band of ``file_path`` into a :class:`RasterLayer`.

    Ports R's ``getRaster`` (without the AOI/mask clipping, which is handled
    separately by :mod:`masking`).
    """
    logger.info(f"Reading raster: {file_path}")
    try:
        with rio.open(file_path) as src:
            array = src.read()
            nodata = na_value if na_value is not None else src.nodata
            profile = _open_profile(src, nodata)
            band_names = [d or f"band_{i + 1}" for i, d in enumerate(src.descriptions)]
            return RasterLayer(array=array, profile=profile, band_names=band_names)
    except Exception:
        logger.error(f"Failed to read {file_path}")
        raise


def read_categorical_raster(file_path: str, na_value: Optional[float] = None) -> RasterLayer:
    """Read a single-band categorical (class-code) raster.

    Ports R's ``getRaster(..., with.single.layer=FALSE)`` for the LULC case:
    a plain single-band layer, kept in its native integer dtype (never
    upcast to float for NaN-encoding of nodata — nodata is tracked via
    ``profile['nodata']`` and :meth:`RasterLayer.valid_mask` instead).
    """
    layer = read_raster(file_path, na_value=na_value)
    if layer.count != 1:
        raise ValueError(f"{file_path} is not single-band ({layer.count} bands found)")
    return layer


def read_one_hot(
    file_path: str,
    class_ids: Sequence[int],
    class_names: Optional[Sequence[str]] = None,
    na_value: Optional[float] = None,
) -> RasterLayer:
    """Read a categorical raster and expand it to one binary band per class.

    Ports R's ``singleToMultiBand`` used inside ``getRaster(with.single.layer=TRUE)``.
    """
    base = read_categorical_raster(file_path, na_value=na_value)
    codes = base.single_band
    bands = np.stack([(codes == cid).astype(np.uint8) for cid in class_ids], axis=0)
    profile = base.profile.copy()
    profile["count"] = len(class_ids)
    profile["dtype"] = "uint8"
    names = list(class_names) if class_names is not None else [f"class_{c}" for c in class_ids]
    return RasterLayer(array=bands, profile=profile, band_names=names)


def _assert_aligned(layers: Dict[str, RasterLayer]) -> None:
    """Raise a clear error if driver rasters are not on a common grid.

    The pre-Slice-1 migration concatenated driver DataFrames positionally
    with no alignment check, silently corrupting data on mismatched grids.
    """
    shapes = {name: layer.shape for name, layer in layers.items()}
    if len(set(shapes.values())) > 1:
        raise DatasetValidationError(f"Driver rasters are not aligned to a common grid: {shapes}")


def read_driver_stack(driver_dict: Dict[str, str]) -> RasterLayer:
    """Read multiple single/multi-band driver rasters into one stacked layer.

    Ports the driver-loading half of R's ``getDataTable`` (``with.driver.name``
    naming). Validates that all rasters share the same grid shape instead of
    assuming it (see :func:`_assert_aligned`).
    """
    layers = {name: read_raster(path) for name, path in driver_dict.items()}
    _assert_aligned(layers)

    arrays: List[np.ndarray] = []
    names: List[str] = []
    reference_profile = None
    for name, layer in layers.items():
        if reference_profile is None:
            reference_profile = layer.profile.copy()
        for i in range(layer.count):
            arrays.append(layer.array[i])
            band_label = name if layer.count == 1 else f"{name}_{i + 1}"
            names.append(band_label)

    stacked = np.stack(arrays, axis=0)
    profile = reference_profile.copy()
    profile["count"] = stacked.shape[0]
    return RasterLayer(array=stacked, profile=profile, band_names=names)


def get_number_of_classes(file_path: str, na_value: Optional[float] = None) -> int:
    """Port of R's ``getNumberOfClass``: count of distinct class codes present."""
    layer = read_categorical_raster(file_path, na_value=na_value)
    codes = layer.single_band
    mask = layer.valid_mask()
    return int(np.unique(codes[mask]).size)


def get_cell_count(file_path: str) -> int:
    """Port of R's ``getnoOfCell``: total number of grid cells (rows * cols)."""
    with rio.open(file_path) as src:
        return int(src.height * src.width)


def write_categorical_raster(
    array: np.ndarray,
    reference_profile: dict,
    output_path: str,
    nodata: float = NA_VALUE,
) -> None:
    """Write a single-band categorical array to a GeoTIFF.

    Unlike the pre-Slice-1 ``dataframe_to_raster``, this honors the caller's
    ``nodata`` argument instead of silently substituting the module-global
    ``NA_VALUE``. The reference profile's dtype is kept as-is
    (R's ``writeRaster`` writes the predicted map in the template's float32
    with the template's NA flag; forcing uint8 here would break parity).
    """
    logger.info(f"Writing output to: {output_path}")
    profile = reference_profile.copy()
    dtype = profile.get("dtype", "float32")

    out = np.asarray(array).astype(dtype)
    profile.update(count=1, dtype=dtype, nodata=nodata, driver="GTiff")

    with rio.open(output_path, "w", **profile) as dst:
        dst.write(out[np.newaxis, ...])


def write_layer(layer: RasterLayer, output_path: str) -> None:
    """Write a full (possibly multi-band) :class:`RasterLayer` to a GeoTIFF."""
    logger.info(f"Writing output to: {output_path}")
    profile = layer.profile.copy()
    profile.update(count=layer.count, driver="GTiff")
    with rio.open(output_path, "w", **profile) as dst:
        dst.write(layer.array)
        for i, name in enumerate(layer.band_names, start=1):
            dst.set_band_description(i, name)
