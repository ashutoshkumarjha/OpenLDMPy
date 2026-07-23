"""Tests for the Pontius three-map disagreement decomposition."""

import numpy as np
import pandas as pd
import pytest

from conftest import FIXTURES, has_fixture, load_csv_matrix

from LULC import pontius


def test_proportion_matrix_sums_to_exactly_one():
    for cm in (
        np.array([[10, 2, 0], [1, 20, 3], [0, 1, 15]], dtype=float),
        np.array([[7, 0], [0, 13]], dtype=float),
        np.array([[100, 1, 1], [1, 100, 1], [1, 1, 100]], dtype=float),
    ):
        pmx = pontius.proportion_matrix(cm)
        assert pmx.sum() == pytest.approx(1.0, abs=1e-12)


def test_proportion_matrix_rejects_non_square():
    with pytest.raises(ValueError):
        pontius.proportion_matrix(np.array([[1, 2, 3], [4, 5, 6]], dtype=float))


def test_collapse_to_two_class_preserves_total():
    cm = np.array([[10, 2, 0, 1], [1, 20, 3, 0], [0, 1, 15, 2], [2, 0, 1, 30]], dtype=float)
    for index in range(cm.shape[0]):
        mm = pontius.collapse_to_two_class(cm, index)
        assert mm.shape == (2, 2)
        assert mm.sum() == pytest.approx(cm.sum())


def test_collapse_to_two_class_matches_hand_computation():
    """cm[1,1]=20 is the pivot class; p=20, rsum=sum(row1)=24, csum=sum(col1)=23,
    pdash=sum of the matrix excluding row1/col1=25. R's column-major
    matrix(c(p,csum-p,rsum-p,pdash), nrow=2) fills [[p, rsum-p], [csum-p, pdash]]."""
    cm = np.array([[10, 2, 0], [1, 20, 3], [0, 1, 15]], dtype=float)
    mm = pontius.collapse_to_two_class(cm, 1)
    np.testing.assert_array_equal(mm, [[20, 4], [3, 25]])


def test_kappa_agreement_index_perfect_prediction_has_zero_disagreement(t1_file, t2_file):
    """If the simulated map exactly equals the actual map, every disagreement
    component collapses to 0 and the non-3-map kappa variants all read 1.0
    — the base map (a different, real raster) only feeds the
    ksimulation/ktransition/ktranslocation trio, not these."""
    result = pontius.kappa_agreement_index(
        simulation_file=t2_file, actual_file=t2_file, base_file=t1_file,
    )
    assert result.quantity_disagreement_overall == pytest.approx(0, abs=1e-6)
    assert result.allocation_disagreement_overall == pytest.approx(0, abs=1e-6)
    assert result.exchange_overall == pytest.approx(0, abs=1e-6)
    assert result.shift_overall == pytest.approx(0, abs=1e-6)
    assert result.cumulative_disagreement_overall == pytest.approx(0, abs=1e-6)
    assert result.allocation_agreement_overall == pytest.approx(1.0, abs=1e-6)
    assert result.kstandard_overall == pytest.approx(1.0, abs=1e-6)
    assert result.kallocation_overall == pytest.approx(1.0, abs=1e-6)
    assert result.khistogram_overall == pytest.approx(1.0, abs=1e-6)
    assert result.kquantity_overall == pytest.approx(1.0, abs=1e-6)
    assert result.kno_overall == pytest.approx(1.0, abs=1e-6)


def test_simulated_agreement_given_observed_transition_pmax_at_least_pe(t1_file, t2_file, t3_file):
    """pmax (max possible agreement) can never be less than pe (chance
    agreement) within any stratum — min(a,b) >= a*b for a,b in [0,1] — and
    that ordering survives the nonnegative-weighted sum across strata."""
    sa = pontius.simulated_agreement_given_observed_transition(
        simulation_file=t3_file, actual_file=t2_file, base_file=t1_file,
    )
    assert sa.pmax_overall >= sa.pe_overall - 1e-9
    assert np.all(sa.pmax_classwise >= sa.pe_classwise - 1e-9)
    assert 0 <= sa.pe_overall <= 1
    assert 0 <= sa.pmax_overall <= 1


def test_kappa_agreement_index_real_data_sane_ranges(t1_file, t2_file, t3_file):
    result = pontius.kappa_agreement_index(
        simulation_file=t3_file, actual_file=t2_file, base_file=t1_file,
    )
    n = result.no_of_class
    assert result.proportion_matrix.shape == (n, n)
    assert result.proportion_matrix.sum() == pytest.approx(1.0, abs=1e-6)
    for arr in (
        result.kstandard_classwise, result.kno_classwise, result.klocation_classwise,
        result.kallocation_classwise, result.khistogram_classwise, result.kquantity_classwise,
        result.ksimulation_classwise, result.ktransition_classwise, result.ktranslocation_classwise,
        result.allocation_disagreement_classwise, result.allocation_agreement_classwise,
        result.shift_classwise, result.exchange_classwise, result.quantity_disagreement_classwise,
        result.cumulative_disagreement_classwise,
    ):
        assert arr.shape == (n,)
    assert 0 <= result.allocation_agreement_overall <= 1


def test_to_table_has_one_overall_and_n_classwise_row_per_metric(t1_file, t2_file, t3_file):
    result = pontius.kappa_agreement_index(
        simulation_file=t3_file, actual_file=t2_file, base_file=t1_file,
    )
    table = result.to_table()
    n_metrics = 8
    assert len(table) == n_metrics * (1 + result.no_of_class)
    assert sum(1 for row in table if row["class"] == "OA") == n_metrics


def test_summary_text_mentions_every_overall_metric(t1_file, t2_file, t3_file):
    result = pontius.kappa_agreement_index(
        simulation_file=t3_file, actual_file=t2_file, base_file=t1_file,
    )
    text = result.summary_text()
    for label in (
        "Overall-Allocation Disagreement", "Overall-Quantitative Disagreement",
        "Overall-kstandard", "Overall-kno", "Overall-kallocation", "Overall-khistogram",
        "Overall-kquantity", "Overall-ksimulation", "Overall-ktransition",
        "Overall-ktranslocation",
    ):
        assert label in text


def test_kappa_agreement_index_respects_class_names(t1_file, t2_file, t3_file, class_names):
    result = pontius.kappa_agreement_index(
        simulation_file=t3_file, actual_file=t2_file, base_file=t1_file,
    )
    names = class_names[: result.no_of_class]
    table = result.to_table(class_names=names)
    classwise_labels = {row["class"] for row in table if row["class"] != "OA"}
    assert classwise_labels <= set(names)


@pytest.mark.skipif(not has_fixture("pontius_overall.csv"), reason="R oracle fixture missing")
def test_kappa_agreement_index_overall_matches_r(t1_file, t2_file, t3_file):
    """R oracle: kappa.agreementindex(simulationFile=T3, actualFile=T2,
    baseFile=T1) via tests/r_oracle/generate_fixtures.R's Pontius section."""
    result = pontius.kappa_agreement_index(
        simulation_file=t3_file, actual_file=t2_file, base_file=t1_file,
    )
    expected = pd.read_csv(FIXTURES / "pontius_overall.csv").set_index("name")["value"]

    assert result.kstandard_overall == pytest.approx(expected["kstandard.overall"])
    assert result.kno_overall == pytest.approx(expected["kno.overall"])
    assert result.klocation_overall == pytest.approx(expected["klocation.overall"])
    assert result.kallocation_overall == pytest.approx(expected["kallocation.overall"])
    assert result.khistogram_overall == pytest.approx(expected["khistogram.overall"])
    assert result.kquantity_overall == pytest.approx(expected["kquantity.overall"])
    assert result.ksimulation_overall == pytest.approx(expected["ksimulation.overall"])
    assert result.ktransition_overall == pytest.approx(expected["ktransition.overall"])
    assert result.ktranslocation_overall == pytest.approx(expected["ktranslocation.overall"])
    assert result.allocation_disagreement_overall == pytest.approx(expected["allocationdisagreemnt.overall"])
    assert result.allocation_agreement_overall == pytest.approx(expected["allocationagreemnt.overall"])
    assert result.shift_overall == pytest.approx(expected["shift.overall"])
    assert result.exchange_overall == pytest.approx(expected["exchange.overall"])
    assert result.quantity_disagreement_overall == pytest.approx(expected["quantitydisagreemnt.overall"])
    assert result.cumulative_disagreement_overall == pytest.approx(expected["cumulativedisagreemnt.overall"])


@pytest.mark.skipif(not has_fixture("pontius_classwise.csv"), reason="R oracle fixture missing")
def test_kappa_agreement_index_classwise_matches_r(t1_file, t2_file, t3_file):
    result = pontius.kappa_agreement_index(
        simulation_file=t3_file, actual_file=t2_file, base_file=t1_file,
    )
    expected = pd.read_csv(FIXTURES / "pontius_classwise.csv")

    np.testing.assert_allclose(result.kstandard_classwise, expected["kstandard.classwise"], atol=1e-9)
    np.testing.assert_allclose(result.kno_classwise, expected["kno.classwise"], atol=1e-9)
    np.testing.assert_allclose(result.klocation_classwise, expected["klocation.classwise"], atol=1e-9)
    np.testing.assert_allclose(result.kallocation_classwise, expected["kallocation.classwise"], atol=1e-9)
    np.testing.assert_allclose(result.khistogram_classwise, expected["khistogram.classwise"], atol=1e-9)
    np.testing.assert_allclose(result.kquantity_classwise, expected["kquantity.classwise"], atol=1e-9)
    np.testing.assert_allclose(result.ksimulation_classwise, expected["ksimulation.classwise"], atol=1e-9)
    np.testing.assert_allclose(result.ktransition_classwise, expected["ktransition.classwise"], atol=1e-9)
    np.testing.assert_allclose(result.ktranslocation_classwise, expected["ktranslocation.classwise"], atol=1e-9)
    np.testing.assert_allclose(
        result.allocation_disagreement_classwise, expected["allocationdisagreemnt.classwise"]
    )
    np.testing.assert_allclose(
        result.allocation_agreement_classwise, expected["allocationagreemnt.classwise"]
    )
    np.testing.assert_allclose(result.shift_classwise, expected["shift.classwise"], atol=1e-9)
    np.testing.assert_allclose(result.exchange_classwise, expected["exchange.classwise"], atol=1e-9)
    np.testing.assert_allclose(
        result.quantity_disagreement_classwise, expected["quantitydisagreemnt.classwise"]
    )
    np.testing.assert_allclose(
        result.cumulative_disagreement_classwise, expected["cumulativedisagreemnt.classwise"]
    )


@pytest.mark.skipif(not has_fixture("pontius_proportionmatrix.csv"), reason="R oracle fixture missing")
def test_kappa_agreement_index_proportion_matrix_matches_r(t1_file, t2_file, t3_file):
    result = pontius.kappa_agreement_index(
        simulation_file=t3_file, actual_file=t2_file, base_file=t1_file,
    )
    expected = load_csv_matrix("pontius_proportionmatrix.csv")
    np.testing.assert_allclose(result.proportion_matrix, expected, atol=1e-6)


def test_kappa_agreement_index_without_base_file_leaves_three_metrics_nan(t2_file, t3_file):
    """base_file is optional (the GUI's "Base File
    (Optional)" field). Without one, everything not derived from
    simulated_agreement_given_observed_transition is still computed
    normally; only ksimulation/ktransition/ktranslocation come back NaN."""
    result = pontius.kappa_agreement_index(simulation_file=t3_file, actual_file=t2_file, base_file=None)

    assert not np.isnan(result.kstandard_overall)
    assert not np.isnan(result.kallocation_overall)
    assert np.isnan(result.ksimulation_overall)
    assert np.isnan(result.ktransition_overall)
    assert np.isnan(result.ktranslocation_overall)
    assert np.all(np.isnan(result.ksimulation_classwise))
    assert np.all(np.isnan(result.ktransition_classwise))
    assert np.all(np.isnan(result.ktranslocation_classwise))


@pytest.mark.skipif(not has_fixture("pontius_overall.csv"), reason="R oracle fixture missing")
def test_kappa_agreement_index_without_base_file_matches_r_on_shared_metrics(t2_file, t3_file):
    """The metrics that don't need a base map should be identical whether
    or not one is given — cross-checked against the same R-oracle fixture
    used by test_kappa_agreement_index_overall_matches_r."""
    result = pontius.kappa_agreement_index(simulation_file=t3_file, actual_file=t2_file, base_file=None)
    expected = pd.read_csv(FIXTURES / "pontius_overall.csv").set_index("name")["value"]

    assert result.kstandard_overall == pytest.approx(expected["kstandard.overall"])
    assert result.kallocation_overall == pytest.approx(expected["kallocation.overall"])
    assert result.khistogram_overall == pytest.approx(expected["khistogram.overall"])
    assert result.kquantity_overall == pytest.approx(expected["kquantity.overall"])


def test_r_safe_ratio_true_zero_over_zero_is_one():
    assert pontius._r_safe_ratio_scalar(0.0, 0.0) == 1.0


def test_r_safe_ratio_nonzero_over_zero_is_nan():
    """Genuinely undefined (R: x[is.infinite(x)]=NA) — must stay NaN, not
    get swallowed by the epsilon tolerance meant for near-zero noise."""
    assert np.isnan(pontius._r_safe_ratio_scalar(0.5, 0.0))


def test_r_safe_ratio_tolerates_cross_computation_floating_point_noise():
    """Real bug, pinned exactly: simulation=a real Execute output,
    actual=2005.tif, base=1995.tif produced ktranslocation_classwise[5]
    ("class6") as NaN/blank in the GUI. Root cause: the denominator
    (pmax_classwise - pe_classwise, both from this module's own 12dp-
    rounded computation) landed on exact 0, while the numerator
    (cehatmqml - pe_classwise, cehatmqml computed via the *separate*
    2-map confusion-matrix path) landed on -8.657e-08 instead of exact 0
    — the same true "0/0" case, but split across two independently-rounded
    computations that don't agree on what "zero" looks like in floats."""
    assert pontius._r_safe_ratio_scalar(-8.665700002019605e-08, 0.0) == 1.0


def test_r_safe_ratio_normal_division_is_unaffected():
    assert pontius._r_safe_ratio_scalar(0.3, 0.6) == pytest.approx(0.5)
    result = pontius._r_safe_ratio(np.array([0.3, 0.9]), np.array([0.6, 0.9]))
    np.testing.assert_allclose(result, [0.5, 1.0])


@pytest.mark.skipif(not has_fixture("kappa_scalars.csv"), reason="R oracle fixtures missing")
def test_proportion_matrix_matches_r_generatepraportionmatrix():
    """Cross-check proportion_matrix's normalization/truncation-error
    behavior against real R-produced counts (a different, already-existing
    fixture — tm_1985_1995.csv — not the Pontius-specific ones above)."""
    cm = load_csv_matrix("tm_1985_1995.csv")
    pmx = pontius.proportion_matrix(cm)
    assert pmx.sum() == pytest.approx(1.0, abs=1e-9)
    np.testing.assert_allclose(pmx, cm / cm.sum(), atol=1e-6)
