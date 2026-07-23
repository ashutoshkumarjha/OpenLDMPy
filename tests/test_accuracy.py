"""R-oracle comparison for accuracy.py."""

import numpy as np
import pandas as pd
import pytest
import rasterio

from conftest import has_fixture, load_csv_matrix, FIXTURES

from LULC import accuracy


@pytest.mark.skipif(not has_fixture("kappa_scalars.csv"), reason="R oracle fixture missing")
def test_kappa_statistics_scalars_match_r():
    tm = load_csv_matrix("tm_1985_1995.csv")
    stats = accuracy.kappa_statistics(tm)
    expected = pd.read_csv(FIXTURES / "kappa_scalars.csv").set_index("name")["value"]

    assert stats.sum_n == pytest.approx(expected["sum.n"])
    assert stats.sum_naive == pytest.approx(expected["sum.naive"])
    assert stats.sum_var == pytest.approx(expected["sum.var"])
    assert stats.sum_kappa == pytest.approx(expected["sum.kappa"])
    assert stats.sum_kvar == pytest.approx(expected["sum.kvar"])


@pytest.mark.skipif(not has_fixture("kappa_per_class.csv"), reason="R oracle fixture missing")
def test_kappa_statistics_per_class_matches_r():
    tm = load_csv_matrix("tm_1985_1995.csv")
    stats = accuracy.kappa_statistics(tm)
    expected = pd.read_csv(FIXTURES / "kappa_per_class.csv")

    np.testing.assert_allclose(stats.user_naive, expected["user.naive"].values)
    np.testing.assert_allclose(stats.prod_naive, expected["prod.naive"].values)
    np.testing.assert_allclose(stats.user_kappa, expected["user.kappa"].values)
    np.testing.assert_allclose(stats.prod_kappa, expected["prod.kappa"].values)


@pytest.mark.skipif(not has_fixture("py_kappa_summary.txt"), reason="R oracle fixture missing")
def test_py_kappa_summary_byte_exact_match_r():
    """The GUI parses this string with a brittle '~~' split; it must match
    R's PyKappasummary output character-for-character."""
    tm = load_csv_matrix("tm_1985_1995.csv")
    stats = accuracy.kappa_statistics(tm)
    expected = (FIXTURES / "py_kappa_summary.txt").read_text().strip()
    assert stats.py_kappa_summary().strip() == expected


def test_kappa_perfect_agreement_is_one():
    cm = np.diag([10, 20, 30])
    stats = accuracy.kappa_statistics(cm)
    assert stats.sum_naive == pytest.approx(1.0)
    assert stats.sum_kappa == pytest.approx(1.0)


def test_is_dataset_correct_true_for_matching_extents(t1_file, t2_file):
    """T1/T2 (both LULC classification rasters, same source pipeline)
    genuinely share extent, NA value, and NA ratio. The bundled sample
    data's *drivers*, however, do not share T2's NA value (see
    test_check_dataset_detects_na_value_mismatch_between_drivers_and_lulc)
    — this test no longer includes them, since asserting True there would
    now correctly be false."""
    assert accuracy.is_dataset_correct({}, {}, t1_file, t2_file) is True


def test_check_dataset_detects_na_value_mismatch_between_drivers_and_lulc(t1_file, t2_file, drivers_85, drivers_95):
    """Real finding in the bundled sample data, not a synthetic fixture:
    the LULC rasters' NA value (-3.3999999521443642e+38) and the driver
    rasters' NA value (-3.4028234663852886e+38, i.e. exactly -FLT_MAX) are
    two different constants — about 0.08% apart, far beyond any floating-
    point-noise tolerance. Since the pipeline can only apply one na_value
    override across all rasters it reads (raster_io.read_raster), no
    single value is correct for both groups; this is exactly the class of
    problem check_dataset's NA-value check exists to surface."""
    report = accuracy.check_dataset(drivers_85, drivers_95, t1_file, t2_file)

    assert report.ok is False
    assert any("NA value" in issue and "DRIVER" in issue for issue in report.issues)
    assert accuracy.is_dataset_correct(drivers_85, drivers_95, t1_file, t2_file) is False


def test_grids_aligned_tolerates_floating_point_noise_but_not_real_shifts():
    """Real-world co-registered rasters can carry sub-micrometer floating
    point noise in their affine origin (observed ~2e-9 map units on the
    bundled sample data) without being genuinely misaligned; a shift on the
    order of a real pixel must still be caught. Exercises _grids_aligned
    directly (no raster I/O) so this stays fast."""
    base = {"height": 100, "width": 100, "crs": "EPSG:32644",
            "transform": rasterio.Affine(200.0, 0.0, 182592.336431867, 0.0, -200.0, 3350002.0)}
    noisy = dict(base, transform=rasterio.Affine(200.0, 0.0, 182592.336431865, 0.0, -200.0, 3350002.0))
    shifted = dict(base, transform=rasterio.Affine(200.0, 0.0, 182592.336431867 + 5000, 0.0, -200.0, 3350002.0))
    different_shape = dict(base, height=50)
    different_crs = dict(base, crs="EPSG:4326")

    assert accuracy._grids_aligned(base, noisy) is True
    assert accuracy._grids_aligned(base, shifted) is False
    assert accuracy._grids_aligned(base, different_shape) is False
    assert accuracy._grids_aligned(base, different_crs) is False


def _write_shifted_copy(src_path, dst_path, dx=5000):
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        data = src.read(1)
    t = profile["transform"]
    profile["transform"] = rasterio.Affine(t.a, t.b, t.c + dx, t.d, t.e, t.f)
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)


def _write_extra_na_copy(src_path, dst_path, size=20):
    with rasterio.open(src_path) as src:
        profile = src.profile.copy()
        data = src.read(1)
    nodata = profile["nodata"]
    valid_rows, valid_cols = np.where(data != nodata)
    r0, c0 = valid_rows[len(valid_rows) // 2], valid_cols[len(valid_cols) // 2]
    data = data.copy()
    data[r0:r0 + size, c0:c0 + size] = nodata
    with rasterio.open(dst_path, "w", **profile) as dst:
        dst.write(data, 1)


def test_check_dataset_detects_extent_mismatch(tmp_path, t1_file, t2_file):
    shifted = tmp_path / "shifted.tif"
    _write_shifted_copy(t1_file, shifted)

    report = accuracy.check_dataset({}, {}, str(shifted), t2_file)

    assert report.ok is False
    assert any("extent does not match" in issue for issue in report.issues)
    assert accuracy.is_dataset_correct({}, {}, str(shifted), t2_file) is False


def test_check_dataset_detects_na_ratio_mismatch(tmp_path, t1_file, t2_file):
    extra_na = tmp_path / "extra_na.tif"
    _write_extra_na_copy(t1_file, extra_na)

    report = accuracy.check_dataset({}, {}, str(extra_na), t2_file)

    assert report.ok is False
    assert any('"LESS"' in issue for issue in report.issues)


def test_check_dataset_extra_layers_checked_like_drivers(tmp_path, t1_file, t2_file):
    shifted = tmp_path / "shifted.tif"
    _write_shifted_copy(t1_file, shifted)

    report = accuracy.check_dataset(
        {}, {}, t1_file, t2_file, extra_layers={"AreaOfInterest": str(shifted)},
    )

    assert report.ok is False
    assert any("AREAOFINTEREST" in issue and "extent does not match" in issue for issue in report.issues)
