"""R-oracle comparison for transition.py (tier 1: exact-value)."""

import numpy as np
import pytest

from conftest import FIXTURES, has_fixture, load_csv_matrix

from LULC import raster_io, transition

DEMAND = [1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992]


@pytest.mark.skipif(not has_fixture("tm_1985_1995.csv"), reason="R oracle fixture missing")
def test_build_transition_matrix_matches_r(t1_file, t2_file):
    t1 = raster_io.read_categorical_raster(t1_file)
    t2 = raster_io.read_categorical_raster(t2_file)
    t1c = np.where(t1.valid_mask(), t1.single_band, np.nan)
    t2c = np.where(t2.valid_mask(), t2.single_band, np.nan)

    tm = transition.build_transition_matrix(t1c, t2c, list(range(1, 10)))
    expected = load_csv_matrix("tm_1985_1995.csv")
    assert np.array_equal(tm, expected)


@pytest.mark.skipif(not has_fixture("tm_1985_1995.csv"), reason="R oracle fixture missing")
def test_get_new_transition_matrix_markov_matches_r():
    tm = load_csv_matrix("tm_1985_1995.csv")
    expected = load_csv_matrix("newtm_markov.csv")
    result = transition.get_new_transition_matrix(tm)
    assert np.array_equal(result, expected)


@pytest.mark.skipif(not has_fixture("newtm_demand.csv"), reason="R oracle fixture missing")
def test_get_new_transition_matrix_with_demand_matches_r():
    tm = load_csv_matrix("tm_1985_1995.csv")
    expected = load_csv_matrix("newtm_demand.csv")
    result = transition.get_new_transition_matrix(tm, demand=DEMAND)
    assert np.array_equal(result, expected)


@pytest.mark.skipif(not has_fixture("yearly_from_newtm_demand.csv"), reason="R oracle fixture missing")
def test_get_yearly_matrix_matches_r():
    """Best-effort comparison (tier 2): R's getYearlyMatrix estimates
    the generator matrix via a hand-rolled truncated series, while this port
    uses scipy.linalg.logm/expm — different numerical methods for the same
    underlying continuous-time-Markov interpolation, so close but not
    byte-exact agreement is expected (see transition.py's module docstring)."""
    new_tm_demand = load_csv_matrix("newtm_demand.csv")
    expected = load_csv_matrix("yearly_from_newtm_demand.csv")
    result = transition.get_yearly_matrix(new_tm_demand, steps=2, for_year=1)
    assert np.abs(result - expected).max() <= 2


def test_get_yearly_matrix_preserves_row_sums():
    tm = load_csv_matrix("newtm_demand.csv") if has_fixture("newtm_demand.csv") else np.array(
        [[10, 2, 0], [1, 20, 3], [0, 1, 15]], dtype=float
    )
    for for_year in (1, 2):
        result = transition.get_yearly_matrix(tm, steps=2, for_year=for_year)
        assert np.allclose(result.sum(axis=1), tm.sum(axis=1))


def test_get_yearly_matrix_full_period_reconstructs_input():
    """for_year == steps should exponentiate the full generator back out,
    reconstructing the original one-step matrix (Steps/for_year now
    genuinely affect the result, unlike before the R-side bug fix)."""
    tm = np.array([[10, 2, 0], [1, 20, 3], [0, 1, 15]], dtype=float)
    result = transition.get_yearly_matrix(tm, steps=2, for_year=2)
    assert np.allclose(result, tm, atol=1)


def test_get_yearly_matrix_intermediate_step_differs_from_full_period():
    """The whole point of the fix: an intermediate step must not equal the
    full-period matrix repeated unchanged (the pre-fix, buggy behavior)."""
    tm = np.array([[10, 2, 0], [1, 20, 3], [0, 1, 15]], dtype=float)
    halfway = transition.get_yearly_matrix(tm, steps=2, for_year=1)
    full = transition.get_yearly_matrix(tm, steps=2, for_year=2)
    assert not np.allclose(halfway, full)


def test_transition_matrix_row_col_sums_conserved():
    """Sanity check independent of R: no cells lost or gained."""
    codes = np.array([[1, 1, 2], [2, 3, 3]], dtype=float)
    codes2 = np.array([[1, 2, 2], [2, 3, 1]], dtype=float)
    tm = transition.build_transition_matrix(codes, codes2, [1, 2, 3])
    assert tm.sum() == codes.size


def test_get_new_transition_matrix_preserves_total_count():
    tm = np.array([[10, 2, 0], [1, 20, 3], [0, 1, 15]], dtype=float)
    result = transition.get_new_transition_matrix(tm)
    assert result.sum() == tm.sum()


def test_redistribute_truncation_error_matches_reference_sums():
    m = np.array([[3.2, 1.1], [0.9, 4.8]])
    ref = np.array([4.0, 6.0])
    out = transition.redistribute_truncation_error(ref, m, "row")
    assert np.isclose(out.sum(axis=1), ref).all()
