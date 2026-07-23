"""Synthetic, hand-computed tests for rasterize.py (no R oracle — rasterise()
has zero R-oracle fixtures)."""

import geopandas as gpd
import pytest
from shapely.geometry import box

from LULC import rasterize
from LULC.errors import RasterizeError


def _write_shp(tmp_path, records, name="classes.shp"):
    gdf = gpd.GeoDataFrame(records, crs="EPSG:32644")
    path = tmp_path / name
    gdf.to_file(path)
    return str(path)


def test_presence_two_disjoint_classes(tmp_path):
    records = {
        "IGBP_CODE": [1, 2],
        "geometry": [box(0, 0, 100, 100), box(100, 0, 200, 100)],
    }
    shp = _write_shp(tmp_path, records)

    layer = rasterize.rasterise(shp, grid_size=50, option="presence")

    assert layer.count == 2
    assert layer.band_names == ["1", "2"]
    band1 = layer.band_by_name("1")
    band2 = layer.band_by_name("2")
    # grid is 4 cols x 2 rows (200/50 x 100/50); class 1 occupies the left
    # two columns, class 2 the right two columns.
    assert band1[:, :2].sum() > 0 and band1[:, 2:].sum() == 0
    assert band2[:, 2:].sum() > 0 and band2[:, :2].sum() == 0


def test_fraction_half_covered_cell(tmp_path):
    # A single polygon covering exactly the left half of one 100x100 cell.
    records = {"IGBP_CODE": [1], "geometry": [box(0, 0, 50, 100)]}
    shp = _write_shp(tmp_path, records)

    layer = rasterize.rasterise(shp, grid_size=100, option="fraction", supersample=20)

    weight = layer.band_by_name("1")[0, 0]
    assert 0.45 < weight < 0.55


def test_fraction_overlap_sums_above_one(tmp_path):
    # Two same-class polygons both fully covering the same cell -> weight ~2.
    records = {
        "IGBP_CODE": [1, 1],
        "IGBP_POLY": ["a", "b"],
        "geometry": [box(0, 0, 100, 100), box(0, 0, 100, 100)],
    }
    shp = _write_shp(tmp_path, records)

    layer = rasterize.rasterise(
        shp, grid_size=100, option="fraction", poly_id_field="IGBP_POLY", supersample=10
    )

    assert layer.band_by_name("1")[0, 0] == pytest.approx(2.0, abs=0.05)


def test_missing_class_field_raises_with_column_list(tmp_path):
    records = {"MaskValue": [1], "geometry": [box(0, 0, 10, 10)]}
    shp = _write_shp(tmp_path, records)

    with pytest.raises(RasterizeError, match="MaskValue"):
        rasterize.rasterise(shp, class_field="IGBP_CODE")


def test_invalid_option_raises(tmp_path):
    records = {"IGBP_CODE": [1], "geometry": [box(0, 0, 10, 10)]}
    shp = _write_shp(tmp_path, records)

    with pytest.raises(RasterizeError):
        rasterize.rasterise(shp, option="bogus")


def test_empty_geometry_raises(tmp_path):
    records = {"IGBP_CODE": [1], "geometry": [None]}
    shp = _write_shp(tmp_path, records)

    with pytest.raises(RasterizeError):
        rasterize.rasterise(shp)
