"""Modeling tests: R-oracle comparison for the logistic branch (deterministic,
tier 2 best-effort) plus structural checks for the other model types."""

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES, has_fixture

from LULC import modeling, raster_io


@pytest.fixture(scope="module")
def t1_arrays(t1_file, drivers_85):
    t1 = raster_io.read_categorical_raster(t1_file)
    t1c = np.where(t1.valid_mask(), t1.single_band, np.nan)
    drv = raster_io.read_driver_stack(drivers_85)
    drv_arr = np.where(drv.valid_mask(), drv.array, np.nan)
    return t1c, drv_arr, drv.band_names


@pytest.mark.skipif(not has_fixture("glm_class1_coefs.csv"), reason="R oracle fixture missing")
def test_logistic_coefficients_close_to_r_glm(t1_arrays, class_names):
    t1c, drv_arr, driver_names = t1_arrays
    class_ids = list(range(1, 10))
    models = modeling.fit_models_separately(
        t1c, drv_arr, driver_names, class_ids, class_names, ["logistic"] * 9
    )
    m = models[0]  # BuildUp
    expected = pd.read_csv(FIXTURES / "glm_class1_coefs.csv").set_index("term")["estimate"]

    assert m.estimator.intercept_[0] == pytest.approx(expected["(Intercept)"], abs=0.05)
    for name in m.driver_names:
        r_coef = expected[f"TD1.{name}"]
        py_coef = dict(zip(m.driver_names, m.estimator.coef_[0]))[name]
        assert py_coef == pytest.approx(r_coef, abs=0.1)

    # Wald-test p-values (statsmodels, display-only — see modeling._fit_logistic_pvalues):
    # one per [intercept] + driver, all valid probabilities.
    assert m.pvalues is not None
    assert len(m.pvalues) == len(m.driver_names) + 1
    assert ((m.pvalues >= 0) & (m.pvalues <= 1)).all()


@pytest.mark.skipif(not has_fixture("glm_class1_suitability_t2.csv"), reason="R oracle fixture missing")
def test_logistic_suitability_close_to_r_predictions(t1_arrays, class_names, drivers_95):
    t1c, drv_arr, driver_names = t1_arrays
    class_ids = list(range(1, 10))
    models = modeling.fit_models_separately(
        t1c, drv_arr, driver_names, class_ids, class_names, ["logistic"] * 9
    )
    drv2 = raster_io.read_driver_stack(drivers_95)
    drv2_arr = np.where(drv2.valid_mask(), drv2.array, np.nan)
    suit = modeling.construct_suitability(models, drv2_arr, drv2.band_names)

    expected = pd.read_csv(FIXTURES / "glm_class1_suitability_t2.csv")
    py_flat = suit["BuildUp"].ravel()
    r_ids = expected["id"].values.astype(int) - 1
    r_weights = expected["weight"].values
    diffs = np.abs(py_flat[r_ids] - r_weights)
    # Best-effort comparison: median absolute difference should be
    # small even though individual cells can diverge more.
    assert np.nanmedian(diffs) < 0.05


def test_suitability_probabilities_in_unit_range(t1_arrays, class_names):
    t1c, drv_arr, driver_names = t1_arrays
    class_ids = list(range(1, 10))
    models = modeling.fit_models_separately(
        t1c, drv_arr, driver_names, class_ids, class_names, ["logistic"] * 9
    )
    suit = modeling.construct_suitability(models, drv_arr, driver_names)
    for grid in suit.values():
        valid = grid[np.isfinite(grid)]
        assert (valid >= 0).all() and (valid <= 1).all()


def test_random_forest_and_svm_branches_run(t1_arrays, class_names):
    t1c, drv_arr, driver_names = t1_arrays
    class_ids = list(range(1, 10))
    model_types = ["randomForest", "svm"] + ["logistic"] * 7
    models = modeling.fit_models_separately(
        t1c, drv_arr, driver_names, class_ids, class_names, model_types
    )
    assert models[0].model_type == "randomForest"
    assert models[1].model_type == "svm"
    # No per-coefficient p-value/significance star for these model types
    # (only logistic/nnet get an auxiliary statsmodels fit).
    assert models[0].pvalues is None
    assert models[1].pvalues is None
    suit = modeling.construct_suitability(models, drv_arr, driver_names)
    assert set(suit.keys()) == set(class_names)


@pytest.mark.parametrize(
    "pvalue, expected",
    [(0.0001, "***"), (0.005, "**"), (0.02, "*"), (0.08, "."), (0.5, "")],
)
def test_significance_star_thresholds(pvalue, expected):
    assert modeling._significance_star(pvalue) == expected


def test_formula_restricts_driver_subset(t1_arrays, class_names):
    t1c, drv_arr, driver_names = t1_arrays
    class_ids = list(range(1, 10))
    formulas = ["T1.BuildUp ~ TD1.Elevation"] + [None] * 8
    models = modeling.fit_models_separately(
        t1c, drv_arr, driver_names, class_ids, class_names, ["logistic"] * 9, model_formulas=formulas
    )
    assert models[0].driver_names == ["Elevation"]
    assert len(models[1].driver_names) == len(driver_names)
