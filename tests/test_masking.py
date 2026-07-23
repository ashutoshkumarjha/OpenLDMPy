"""Masking/AOI tests.

Raster-format mask gets a real R-oracle comparison (tier 1: the
sample Mask2005.tif doesn't depend on the retired rgdal/rgeos/maptools).
Shapefile-format AOI/mask is validated structurally only —
there is no R fixture for it.
"""

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES, has_fixture

from LULC import masking, raster_io


@pytest.mark.skipif(not has_fixture("masked_t2_values.csv"), reason="R oracle fixture missing")
def test_raster_mask_matches_r(t2_file, data_dir):
    mask_file = str(data_dir / "MaksFiles/Mask/Mask2005.tif")
    t2 = raster_io.read_categorical_raster(t2_file)
    masked = masking.apply_mask_and_aoi(t2, mask_file=mask_file)

    flat = masked.single_band.ravel()
    valid = np.isfinite(flat) & (flat != masked.nodata)
    py_ids = np.nonzero(valid)[0]

    expected = pd.read_csv(FIXTURES / "masked_t2_values.csv")
    r_ids = expected["id"].values.astype(int) - 1  # R is 1-based
    r_values = dict(zip(r_ids.tolist(), expected["value"].values.tolist()))

    assert set(py_ids.tolist()) == set(r_ids.tolist())
    mismatches = [i for i in py_ids if flat[i] != r_values[int(i)]]
    assert not mismatches


def test_mask_reduces_valid_cell_count(t2_file, data_dir):
    mask_file = str(data_dir / "MaksFiles/Mask/Mask2005.tif")
    t2 = raster_io.read_categorical_raster(t2_file)
    before = int(t2.valid_mask().sum())
    masked = masking.apply_mask_and_aoi(t2, mask_file=mask_file)
    after = int(masked.valid_mask().sum())
    assert 0 < after < before


def test_aoi_shapefile_keeps_only_covered_cells(t2_file, data_dir):
    """Structural-only check: no R oracle for the shapefile path."""
    aoi_shp = str(data_dir / "MaksFiles/AOI/BigCities.shp")
    t2 = raster_io.read_categorical_raster(t2_file)
    before = int(t2.valid_mask().sum())
    clipped = masking.apply_mask_and_aoi(t2, aoi_file=aoi_shp)
    after = int(clipped.valid_mask().sum())
    assert 0 < after < before


def test_mask_shapefile_excludes_covered_cells(t2_file, data_dir):
    mask_shp = str(data_dir / "MaksFiles/Mask/Mask.shp")
    t2 = raster_io.read_categorical_raster(t2_file)
    before = int(t2.valid_mask().sum())
    masked = masking.apply_mask_and_aoi(t2, mask_file=mask_shp)
    after = int(masked.valid_mask().sum())
    assert 0 < after < before


def test_no_op_without_mask_or_aoi(t2_file):
    t2 = raster_io.read_categorical_raster(t2_file)
    same = masking.apply_mask_and_aoi(t2)
    assert same is t2
