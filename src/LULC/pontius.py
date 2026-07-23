"""Pontius three-map disagreement decomposition (originally deferred,
migrated later).

Ports, verbatim in formula (transcribed from Rasterise_dev_68akj.r
lines 1522-1878):

* ``genratePraportionMatrix``           -> :func:`proportion_matrix`
* ``nMatrixto2Class``                   -> :func:`collapse_to_two_class`
* ``getSAGivenOTrans``                  -> :func:`simulated_agreement_given_observed_transition`
* ``kappa.agreementindex``              -> :func:`kappa_agreement_index`
* ``summary.kappa.agreementindex``      -> :meth:`PontiusAgreementIndex.summary_text`
* ``summary.kappa.agreementindex.tabel``-> :meth:`PontiusAgreementIndex.to_table`

This decomposes agreement between a **simulated** map and the **actual**
map it's meant to predict into components (quantity vs. allocation
disagreement, exchange vs. shift, and — the part that needs a third,
**base** map — how much of the agreement is attributable to correctly
predicting *which pixels would change at all*, conditioned on the known
base-year state) per Pontius (2000, 2011), Pontius & Millones (2014), and
Van Vliet et al. (2011). It's a superset of the simpler ``kappa``/
``PyKappasummary`` chain already in ``accuracy.py``, which only needs two
maps and is what the GUI's Accuracy Assessment tab's Confusion
Matrix/Predicted Accuracy/Reference Reliability views use.

``getUnbiasedEstimatePropertionMatrix`` (R lines 1499-1520) is intentionally
**not** ported: it references an undefined R variable (``noLC``) and has 0
call sites anywhere in the R source, including within this very function
family — dead, broken code, same category as the other intentionally
un-ported omissions elsewhere in this migration.

GUI-wired via the Accuracy Assessment tab's "Base File (Optional)" field
and Agreement Index Classwise/Overall radio buttons (``OpenLDMgui.ui``);
``base_file`` is optional here for exactly that reason — see
:func:`kappa_agreement_index`'s docstring for what's skipped without one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

from .config import logger
from .raster_io import read_categorical_raster
from .transition import _which_max_colmajor, build_transition_matrix


def proportion_matrix(cm: np.ndarray) -> np.ndarray:
    """Port of R's ``genratePraportionMatrix``: normalize a square
    confusion/transition matrix to proportions summing to exactly 1.0,
    nudging the single largest cell (column-major tie-break, matching R's
    ``which.max``) to absorb rounding truncation error."""
    cmx = np.asarray(cm, dtype=float)
    if cmx.ndim != 2 or cmx.shape[0] != cmx.shape[1]:
        raise ValueError("Confusion matrix is not square")
    pmx = np.round(cmx / cmx.sum(), 7)
    r, c = _which_max_colmajor(pmx)
    pmx[r, c] += round(1 - pmx.sum(), 7)
    return pmx


def collapse_to_two_class(m: np.ndarray, index: int) -> np.ndarray:
    """Port of R's ``nMatrixto2Class``: collapse an NxN matrix into a 2x2
    "class ``index`` vs. everything else" matrix, R's own row/column-major
    fill order preserved (``[[p, rsum-p], [csum-p, pdash]]`` — *not*
    ``[[p, csum-p], [rsum-p, pdash]]``, which is what a naive transcription
    of the source order would give; R's ``matrix(c(p,csum-p,rsum-p,pdash),
    nrow=2,ncol=2)`` fills column-major). ``index`` is 0-based (R's is
    1-based)."""
    m = np.asarray(m, dtype=float)
    p = m[index, index]
    rsum = m[index, :].sum()
    csum = m[:, index].sum()
    keep = np.ones(m.shape[0], dtype=bool)
    keep[index] = False
    pdash = m[np.ix_(keep, keep)].sum()
    return np.array([[p, rsum - p], [csum - p, pdash]])


@dataclass
class SAGivenOTransResult:
    """Port of R's ``getSAGivenOTrans`` return list."""

    pe_overall: float
    pe_classwise: np.ndarray
    pmax_overall: float
    pmax_classwise: np.ndarray
    tp_classwise: np.ndarray  # conditionalTP: [simulated, actual, base] counts


def simulated_agreement_given_observed_transition(
    simulation_file: str,
    actual_file: str,
    base_file: str,
    na_value: Optional[float] = None,
) -> SAGivenOTransResult:
    """Port of R's ``getSAGivenOTrans``.

    For each base-year class stratum, estimates the chance-agreement
    (``pe``) and maximum-possible-agreement (``pmax``) between the
    simulated and actual maps *within that stratum*, then combines strata
    weighted by their share of the total pixel count — both overall and
    per-class (via :func:`collapse_to_two_class`).

    R builds a single combined raster (``base*B^2 + simulated*B +
    actual``) and tabulates it with ``freq()`` to get the 3-way joint
    counts; this reads the three rasters directly and bins the
    (simulated, actual, base) triplets with ``np.bincount`` instead —
    same result, without the arithmetic-encoding trick R needed to reuse
    its single-raster ``freq()`` helper.

    Deviation from R: a base-class stratum with zero pixels produces
    ``0/0`` in R (propagating ``NaN`` through the *entire* weighted sum
    below, even though a zero-weight stratum should contribute nothing).
    This implementation treats a zero-count stratum as contributing 0 to
    both ``pe`` and ``pmax`` instead — the stratum's weight
    (``sa_prop[o]``) is 0 either way, so this only avoids an
    otherwise-spurious ``NaN`` from a stratum that occurs 0 times in this
    particular dataset.
    """
    base = read_categorical_raster(base_file, na_value=na_value)
    actual = read_categorical_raster(actual_file, na_value=na_value)
    simulated = read_categorical_raster(simulation_file, na_value=na_value)

    valid = base.valid_mask() & actual.valid_mask() & simulated.valid_mask()
    b = base.single_band[valid]
    a = actual.single_band[valid]
    s = simulated.single_band[valid]

    class_ids = np.array(sorted(set(np.unique(b)) | set(np.unique(a)) | set(np.unique(s))))
    n = class_ids.size
    bi = np.searchsorted(class_ids, b)
    ai = np.searchsorted(class_ids, a)
    si = np.searchsorted(class_ids, s)

    # conditional_tp[simulated, actual, base] = joint pixel count
    flat = (si * n + ai) * n + bi
    conditional_tp = np.bincount(flat, minlength=n * n * n).reshape(n, n, n).astype(np.int64)

    total = conditional_tp.sum()
    sa_prop = np.zeros(n)
    pe = np.zeros((n, n))       # [stratum, class]
    pmax = np.zeros((n, n))
    c_pe = np.zeros((n, n))     # [class, stratum]
    c_pmax = np.zeros((n, n))

    for o in range(n):
        stratum_total = conditional_tp[:, :, o].sum()
        if stratum_total == 0:
            continue  # sa_prop[o] stays 0; pe/pmax rows stay 0 (see docstring)
        sa_prop[o] = stratum_total / total
        est = conditional_tp[:, :, o] / stratum_total
        row_sums = est.sum(axis=1)
        col_sums = est.sum(axis=0)
        pe[o, :] = row_sums * col_sums
        pmax[o, :] = np.minimum(row_sums, col_sums)

        for cls in range(n):
            mm = collapse_to_two_class(est, cls)
            rs, cs = mm.sum(axis=1), mm.sum(axis=0)
            c_pe[cls, o] = np.sum(rs * cs)
            c_pmax[cls, o] = np.sum(np.minimum(rs, cs))

    # Rounded to 12dp, matching this codebase's/R's own pervasive round(...,7)
    # truncation-error guards: a value that's mathematically exactly 1.0 can
    # land a few ULPs away (e.g. 0.9999999999999999) purely from
    # floating-point summation order, which would wrongly dodge the
    # ksimulation/ktransition/ktranslocation 0/0-is-1 special case below for
    # a denominator that's "supposed to" be exactly 0.
    pe_overall = round(float((pe.sum(axis=1) * sa_prop).sum()), 12)
    pmax_overall = round(float((pmax.sum(axis=1) * sa_prop).sum()), 12)
    pe_classwise = np.round((c_pe * sa_prop).sum(axis=1), 12)
    pmax_classwise = np.round((c_pmax * sa_prop).sum(axis=1), 12)

    return SAGivenOTransResult(
        pe_overall=pe_overall,
        pe_classwise=pe_classwise,
        pmax_overall=pmax_overall,
        pmax_classwise=pmax_classwise,
        tp_classwise=conditional_tp,
    )


_R_NA_EPS = 1e-6  # see _r_safe_ratio


def _r_safe_ratio(numerator, denominator):
    """R: ``x[is.na(x)]=1; x[is.infinite(x)]=NA`` for ``x = numerator /
    denominator`` — done here with an epsilon tolerance on *both* operands
    rather than relying on bit-exact 0/0, because numerator and denominator
    can come from independently-computed quantities that each land a few
    ULPs off zero even when the true underlying value is genuinely
    degenerate/zero (observed: a rare class where the denominator, built
    from this module's own 12dp-rounded pe/pmax, lands on exact 0, while
    the numerator — built from the separately-computed confusion-matrix
    path — lands on ``-8.67e-08`` instead of exact 0; R, doing everything
    in one computation, wouldn't see that split). 1e-6 is far below any
    real class-level difference for realistic pixel counts, so this only
    catches genuinely-degenerate cases, not real small-but-nonzero ones."""
    numerator = np.asarray(numerator, dtype=float)
    denominator = np.asarray(denominator, dtype=float)
    near_zero_num = np.abs(numerator) < _R_NA_EPS
    near_zero_denom = np.abs(denominator) < _R_NA_EPS
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = numerator / denominator
    ratio = np.where(near_zero_denom & near_zero_num, 1.0, ratio)
    ratio = np.where(near_zero_denom & ~near_zero_num, np.nan, ratio)
    return ratio


def _r_safe_ratio_scalar(numerator: float, denominator: float) -> float:
    return float(_r_safe_ratio(numerator, denominator))


@dataclass
class PontiusAgreementIndex:
    """Port of R's ``kappa.agreementindex`` return list. ``*_classwise``
    fields are arrays of length ``no_of_class``; ``*_overall`` fields are
    scalars."""

    no_of_class: int
    proportion_matrix: np.ndarray

    kstandard_classwise: np.ndarray
    kstandard_overall: float
    kno_classwise: np.ndarray
    kno_overall: float
    klocation_classwise: np.ndarray
    klocation_overall: float
    kallocation_classwise: np.ndarray
    kallocation_overall: float
    khistogram_classwise: np.ndarray
    khistogram_overall: float
    kquantity_classwise: np.ndarray
    kquantity_overall: float
    ksimulation_classwise: np.ndarray
    ksimulation_overall: float
    ktransition_classwise: np.ndarray
    ktransition_overall: float
    ktranslocation_classwise: np.ndarray
    ktranslocation_overall: float

    allocation_disagreement_overall: float
    allocation_agreement_overall: float
    shift_overall: float
    exchange_overall: float
    quantity_disagreement_overall: float
    cumulative_disagreement_overall: float
    allocation_disagreement_classwise: np.ndarray
    allocation_agreement_classwise: np.ndarray
    shift_classwise: np.ndarray
    exchange_classwise: np.ndarray
    quantity_disagreement_classwise: np.ndarray
    cumulative_disagreement_classwise: np.ndarray

    def summary_text(self, class_names: Optional[Sequence[str]] = None) -> str:
        """Port of R's ``summary.kappa.agreementindex`` console report."""
        names = list(class_names) if class_names is not None else [
            f"Class{i + 1}" for i in range(self.no_of_class)
        ]

        def cw(label: str, arr: np.ndarray) -> str:
            pairs = ", ".join(f"{n}={round(float(v), 7)}" for n, v in zip(names, arr))
            return f"classwise-{label}: {pairs}"

        lines = [
            "Propertionmatrix:",
            np.array2string(np.round(self.proportion_matrix, 7)),
            cw("Allocation Disagreement", self.allocation_disagreement_classwise),
            cw("Quantitative Disagreement", self.quantity_disagreement_classwise),
            cw("Agreement", self.allocation_agreement_classwise),
            cw("Cumulative Disagreement", self.cumulative_disagreement_classwise),
            cw("Shift", self.shift_classwise),
            cw("Exchange", self.exchange_classwise),
            f"Overall-Allocation Disagreement: {round(self.allocation_disagreement_overall, 7)}",
            f"Overall-Quantitative Disagreement: {round(self.quantity_disagreement_overall, 7)}",
            f"Overall-Agreement: {round(self.allocation_agreement_overall, 7)}",
            f"Overall-Cumulative Disagreement: {round(self.cumulative_disagreement_overall, 7)}",
            f"Overall-kstandard: {round(self.kstandard_overall, 7)}",
            f"Overall-kno: {round(self.kno_overall, 7)}",
            f"Overall-kallocation: {round(self.kallocation_overall, 7)}",
            f"Overall-khistogram: {round(self.khistogram_overall, 7)}",
            f"Overall-kquantity: {round(self.kquantity_overall, 7)}",
            f"Overall-ksimulation: {round(self.ksimulation_overall, 7)}",
            f"Overall-ktransition: {round(self.ktransition_overall, 7)}",
            f"Overall-ktranslocation: {round(self.ktranslocation_overall, 7)}",
            cw("kstandard", self.kstandard_classwise),
            cw("kno", self.kno_classwise),
            cw("kallocation", self.kallocation_classwise),
            cw("khistogram", self.khistogram_classwise),
            cw("kquantity", self.kquantity_classwise),
            cw("ksimulation", self.ksimulation_classwise),
            cw("ktransition", self.ktransition_classwise),
            cw("ktranslocation", self.ktranslocation_classwise),
        ]
        return "\n".join(lines)

    def to_table(self, class_names: Optional[Sequence[str]] = None) -> List[Dict[str, object]]:
        """Port of R's ``summary.kappa.agreementindex.tabel``: a flat list
        of ``{"kappa": value, "kappa_type": name, "class": label}`` rows —
        one "OA" (overall) row per metric, then ``no_of_class`` classwise
        rows per metric. R returns a character matrix (Kappa, Kappa Type)
        with rownames "OA"/classnames; this is the same information as a
        list of dicts, easier to consume from Python than a matrix of
        stringified numbers."""
        names = list(class_names) if class_names is not None else [
            f"Class{i + 1}" for i in range(self.no_of_class)
        ]
        metrics = [
            ("Kstandard", self.kstandard_overall, self.kstandard_classwise),
            ("Kno", self.kno_overall, self.kno_classwise),
            ("Kallocation", self.kallocation_overall, self.kallocation_classwise),
            ("Khistogram", self.khistogram_overall, self.khistogram_classwise),
            ("Kquantity", self.kquantity_overall, self.kquantity_classwise),
            ("Ksimulation", self.ksimulation_overall, self.ksimulation_classwise),
            ("Ktransition", self.ktransition_overall, self.ktransition_classwise),
            ("Ktranslocation", self.ktranslocation_overall, self.ktranslocation_classwise),
        ]
        rows: List[Dict[str, object]] = []
        for kappa_type, overall, _classwise in metrics:
            rows.append({"kappa": round(float(overall), 7), "kappa_type": kappa_type, "class": "OA"})
        for kappa_type, _overall, classwise in metrics:
            for name, value in zip(names, classwise):
                rows.append({"kappa": round(float(value), 7), "kappa_type": kappa_type, "class": name})
        return rows


def kappa_agreement_index(
    simulation_file: str,
    actual_file: str,
    base_file: Optional[str] = None,
    na_value: Optional[float] = None,
) -> PontiusAgreementIndex:
    """Port of R's ``kappa.agreementindex``: the full Pontius/Van Vliet
    disagreement decomposition comparing a simulated map against the
    actual map it was meant to predict, conditioned on the known
    ``base_file`` state (Pontius 2000, 2011; Pontius & Millones 2014;
    Van Vliet et al. 2011 — see inline formula citations below, transcribed
    from the R source's own comments).

    ``base_file`` is optional here — R's original always required it, but
    the GUI treats the base map as an optional third input (the "Optional
    Base File" field on the Accuracy Assessment tab). When
    omitted, everything that doesn't depend on
    :func:`simulated_agreement_given_observed_transition` is still computed
    normally; ``ksimulation``/``ktransition``/``ktranslocation`` (both
    ``_classwise`` and ``_overall``) are set to ``NaN`` instead, since
    they're the only metrics that need a third map at all.
    """
    logger.info("--- Calculating Pontius agreement/disagreement decomposition ---")
    simulated = read_categorical_raster(simulation_file, na_value=na_value)
    actual = read_categorical_raster(actual_file, na_value=na_value)
    valid = simulated.valid_mask() & actual.valid_mask()
    sim_codes = np.where(valid, simulated.single_band, np.nan)
    act_codes = np.where(valid, actual.single_band, np.nan)
    ids = sorted(
        set(np.unique(sim_codes[np.isfinite(sim_codes)]).astype(int))
        | set(np.unique(act_codes[np.isfinite(act_codes)]).astype(int))
    )
    cm1 = build_transition_matrix(sim_codes, act_codes, ids)  # rows=simulation, cols=actual

    pmx = proportion_matrix(cm1)
    no_lc = pmx.shape[1]
    csumpmx = pmx.sum(axis=0)
    rsumpmx = pmx.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        cp = np.diag(pmx)  # Correct proportion — checked/expected agreement (Pontius 2011)
        qc = np.abs(csumpmx - rsumpmx)  # Quantitative disagreement (Pontius 2011)
        ac = 2 * (np.minimum(csumpmx, rsumpmx) - cp)  # Max allocation disagreement (Pontius 2011)
        dc = csumpmx + rsumpmx - 2 * cp  # Cumulative disagreement, classwise (Pontius 2014 eq1)

        ec = 2 * (np.minimum(pmx, pmx.T).sum(axis=0) - cp)  # Exchange (Pontius 2014 eq4)
        sc = dc - qc - ec  # Shift (Pontius 2014 eq5)
        total_a = ac.sum() / 2  # Total allocation disagreement (Pontius 2011 eq5)
        total_c = cp.sum()  # Total correct proportion (Pontius 2011 eq6)
        total_q = qc.sum() / 2  # Total quantitative disagreement (Pontius 2014 eq6)
        total_e = ec.sum() / 2  # Total exchange (Pontius 2014 eq7)
        total_s = sc.sum() / 2  # Total shift (Pontius 2014 eq8)
        total_d = dc.sum() / 2  # Total disagreement due to allocation (Pontius 2014 eq9 / 2011 eq7)

        ehatnqnl = np.full(no_lc, 1 / no_lc)
        ehatnqpl = np.minimum(1 / no_lc, csumpmx)  # Pontius 2000 Table 2
        ehatmqnl = csumpmx * rsumpmx  # Pontius 2000 Table 2
        ehatmqpl = np.minimum(csumpmx, rsumpmx)  # Pontius 2000 Table 2
        ehatpqnl = csumpmx * csumpmx  # Pontius 2000 Table 2

        cehatmqml = 1 - dc  # equals cp for the two-class case
        cehatnqpl = np.minimum(1 / no_lc, csumpmx) + np.minimum(1 / no_lc, 1 - csumpmx)
        cehatmqnl = csumpmx * rsumpmx + (1 - csumpmx) * (1 - rsumpmx)
        cehatpqnl = csumpmx * csumpmx + (1 - csumpmx) * (1 - csumpmx)

        kstandard = (cehatmqml - cehatmqnl) / (1 - cehatmqnl)
        Kstandard = (total_c - ehatmqnl.sum()) / (1 - ehatmqnl.sum())  # Pontius 2011 eq11

        klocation = (cp - ehatmqnl) / (ehatmqpl - ehatmqnl)  # consistent with Van Vliet 2011 eq5
        Klocation = (total_c - ehatmqnl.sum()) / (ehatmqpl.sum() - ehatmqnl.sum())  # Van Vliet 2011 eq6

        khistogram = (1 - qc - cehatmqnl) / (1 - cehatmqnl)
        Khistogram = (1 - total_q - ehatmqnl.sum()) / (1 - ehatmqnl.sum())

        kno = (cehatmqml - ehatnqnl) / (1 - ehatnqnl)
        Kno = (total_c - 1 / no_lc) / (1 - 1 / no_lc)

        kallocation = (cehatmqml - cehatmqnl) / (1 - qc - cehatmqnl)
        Kallocation = (total_c - ehatmqnl.sum()) / (1 - total_q - ehatmqnl.sum())

        cehatpqml = cehatpqnl + kallocation * (1 - cehatpqnl)
        cehatnqml = ehatnqnl + kallocation * (cehatnqpl - ehatnqnl)

        y = cehatpqml
        z = cehatnqml
        Y = ehatpqnl.sum() + Kallocation * (1 - ehatpqnl.sum())
        Z = 1 / no_lc + Kallocation * (ehatnqpl.sum() - 1 / no_lc)
        kquantity = (cehatmqml - z) / (y - z)
        Kquantity = (total_c - Z) / (Y - Z)

        if base_file:
            sa = simulated_agreement_given_observed_transition(
                simulation_file, actual_file, base_file, na_value
            )

            ksimulation = _r_safe_ratio(cehatmqml - sa.pe_classwise, 1 - sa.pe_classwise)
            Ksimulation = _r_safe_ratio_scalar(total_c - sa.pe_overall, 1 - sa.pe_overall)

            ktransition = _r_safe_ratio(sa.pmax_classwise - sa.pe_classwise, 1 - sa.pe_classwise)
            Ktransition = _r_safe_ratio_scalar(sa.pmax_overall - sa.pe_overall, 1 - sa.pe_overall)

            ktranslocation = _r_safe_ratio(
                cehatmqml - sa.pe_classwise, sa.pmax_classwise - sa.pe_classwise
            )
            Ktranslocation = _r_safe_ratio_scalar(
                total_c - sa.pe_overall, sa.pmax_overall - sa.pe_overall
            )
        else:
            # No base map given — the three metrics that need one are simply
            # not computable (see the function docstring), not R behavior to
            # replicate; NaN is this port's own "not computed" marker.
            ksimulation = np.full(no_lc, np.nan)
            Ksimulation = float("nan")
            ktransition = np.full(no_lc, np.nan)
            Ktransition = float("nan")
            ktranslocation = np.full(no_lc, np.nan)
            Ktranslocation = float("nan")

    return PontiusAgreementIndex(
        no_of_class=no_lc,
        proportion_matrix=pmx,
        kstandard_classwise=kstandard, kstandard_overall=Kstandard,
        kno_classwise=kno, kno_overall=Kno,
        klocation_classwise=klocation, klocation_overall=Klocation,
        kallocation_classwise=kallocation, kallocation_overall=Kallocation,
        khistogram_classwise=khistogram, khistogram_overall=Khistogram,
        kquantity_classwise=kquantity, kquantity_overall=Kquantity,
        ksimulation_classwise=ksimulation, ksimulation_overall=Ksimulation,
        ktransition_classwise=ktransition, ktransition_overall=Ktransition,
        ktranslocation_classwise=ktranslocation, ktranslocation_overall=Ktranslocation,
        allocation_disagreement_overall=total_a, allocation_agreement_overall=total_c,
        shift_overall=total_s, exchange_overall=total_e,
        quantity_disagreement_overall=total_q, cumulative_disagreement_overall=total_d,
        allocation_disagreement_classwise=ac, allocation_agreement_classwise=cp,
        shift_classwise=sc, exchange_classwise=ec,
        quantity_disagreement_classwise=qc, cumulative_disagreement_classwise=dc,
    )
