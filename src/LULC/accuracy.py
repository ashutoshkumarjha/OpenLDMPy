"""Accuracy assessment: confusion matrix, Cohen's kappa family, dataset checks.

Ports:

* ``kappa``            -> :func:`kappa_statistics` (formulas transcribed
  verbatim from Rasterise_dev_68akj.r lines 1456-1497)
* ``summary.kappa``    -> :meth:`KappaStatistics.summary_text`
* ``PyKappasummary``   -> :meth:`KappaStatistics.py_kappa_summary` (the exact
  ``~~``-delimited string the GUI's Accuracy Assessment tab parses)
* ``getKappaSummary``  -> :func:`LULCAlgorithms.get_kappa_summary` (facade)
* ``isDataSetCorrect`` -> :func:`is_dataset_correct`

The confusion matrix itself is R's ``createTM`` applied to the actual and
predicted maps — reuse :func:`transition.build_transition_matrix`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import norm

from .config import logger
from .raster_io import read_raster


@dataclass
class KappaStatistics:
    """All statistics returned by R's ``kappa(CM)`` list, same field meanings."""

    sum_n: float
    sum_naive: float          # overall accuracy (theta1)
    sum_var: float
    sum_kappa: float
    sum_kvar: float
    user_naive: np.ndarray    # per-class d/rowsum
    prod_naive: np.ndarray    # per-class d/colsum
    user_kappa: np.ndarray
    user_kvar: np.ndarray
    prod_kappa: np.ndarray
    prod_kvar: np.ndarray

    def _ciw(self, var: float, alpha: float) -> float:
        # R: qnorm(1-(alpha/2))*sqrt(var) + (1/(2*n))
        return float(norm.ppf(1 - alpha / 2) * np.sqrt(var) + 1 / (2 * self.sum_n))

    def confidence_interval(self, which: str = "kappa", alpha: float = 0.05) -> tuple:
        if which == "kappa":
            center, var = self.sum_kappa, self.sum_kvar
        else:
            center, var = self.sum_naive, self.sum_var
        w = self._ciw(var, alpha)
        return (center - w, center + w)

    def summary_text(self, alpha: float = 0.05) -> str:
        """Port of R's ``summary.kappa`` console report."""
        acc_lo, acc_hi = self.confidence_interval("accuracy", alpha)
        kap_lo, kap_hi = self.confidence_interval("kappa", alpha)
        pct = round((1 - alpha) * 100)
        lines = [
            f"Number of observations: {self.sum_n:.0f}",
            "Summary of naive statistics",
            f"Overall accuracy, stdev, CV%: {round(self.sum_naive, 4)} , "
            f"{round(float(np.sqrt(self.sum_var)), 4)} , "
            f"{round((np.sqrt(self.sum_var) / self.sum_naive) * 1000) / 10}",
            f"{pct} % confidence limits for accuracy: {round(acc_lo, 4)} ... {round(acc_hi, 4)}",
            f"User's accuracy: {np.round(self.user_naive, 4).tolist()}",
            f"Producer's reliability: {np.round(self.prod_naive, 4).tolist()}",
            "Summary of kappa statistics",
            f"Overall kappa, stdev, & CV%: {round(self.sum_kappa, 4)} , "
            f"{round(float(np.sqrt(self.sum_kvar)), 4)} , "
            f"{round((np.sqrt(self.sum_kvar) / self.sum_kappa) * 1000) / 10}",
            f"{pct} % confidence limits for kappa: {round(kap_lo, 4)} ... {round(kap_hi, 4)}",
        ]
        return "\n".join(lines)

    def py_kappa_summary(self) -> str:
        """Port of R's ``PyKappasummary``: the ``~~``-delimited string the GUI
        splits into its Accuracy Assessment fields. Field order and labels
        must not change."""
        w1 = self._ciw(self.sum_var, 0.05)
        w2 = self._ciw(self.sum_var, 0.01)
        # R: paste(round(x, 4), collapse=",") — trailing zeros dropped ("1",
        # not "1.0000"), which %g reproduces.
        uac = ",".join(f"{round(float(v), 4):g}" for v in self.user_naive)
        pac = ",".join(f"{round(float(v), 4):g}" for v in self.prod_naive)
        return (
            f"NoOfObservation:{self.sum_n:.0f}~~"
            f"OverallOfAccuracy:{self.sum_naive:.4f}~~"
            f"95CI:{self.sum_naive - w1:.4f}-{self.sum_naive + w1:.4f}~~"
            f"99CI:{self.sum_naive - w2:.4f}-{self.sum_naive + w2:.4f}~~"
            f"UserAccuracy:{uac}~~"
            f"ProducerReliability:{pac}~~"
            f"Overallkappa:{self.sum_kappa:.4f}~~"
            f"95CI:{self.sum_kappa - w1:.4f}-{self.sum_kappa + w1:.4f}~~"
            f"95CI:{self.sum_kappa - w2:.4f}-{self.sum_kappa + w2:.4f}~~"
        )


def kappa_statistics(cm: np.ndarray) -> KappaStatistics:
    """Cohen's kappa and per-class statistics from a confusion matrix.

    Verbatim port of R's ``kappa(CM)`` — rows are the first map passed to
    createTM (actual), columns the second (predicted).
    """
    cmx = np.asarray(cm, dtype=float)
    if cmx.shape[0] != cmx.shape[1]:
        raise ValueError("Confusion matrix is not square")

    n = cmx.sum()
    d = np.diag(cmx)
    th1 = d.sum() / n
    th1v = th1 * (1 - th1) / n
    csum = cmx.sum(axis=0)
    rsum = cmx.sum(axis=1)

    with np.errstate(divide="ignore", invalid="ignore"):
        ua = d / rsum
        pa = d / csum

        th2 = float(rsum @ csum) / n**2
        kh = (th1 - th2) / (1 - th2)
        th3 = float(((csum + rsum) * d).sum()) / n**2
        # R: th4 accumulates cmx[i,j] * (csum[i] + rsum[j])^2 (note the
        # transposed marginals — csum indexed by row, rsum by column).
        th4 = float((cmx * (csum[:, None] + rsum[None, :]) ** 2).sum()) / n**3
        th1c, th2c = 1 - th1, 1 - th2
        khv = (
            (th1 * th1c) / th2c**2
            + (2 * th1c * (2 * th1 * th2 - th3)) / th2c**3
            + (th1c**2 * (th4 - 4 * th2**2)) / th2c**4
        ) / n

        p = cmx / n
        uap = p.sum(axis=1)
        pap = p.sum(axis=0)
        dp = np.diag(p)
        kpu = (dp / uap - pap) / (1 - pap)
        t1 = uap - dp
        t2 = pap * uap - dp
        t3 = dp * (1 - uap - pap + dp)
        kpuv = ((t1 / (uap**3 * (1 - pap) ** 3)) * (t1 * t2 + t3)) / n
        kpp = (dp / pap - uap) / (1 - uap)
        t1p = pap - dp
        kppv = ((t1p / (pap**3 * (1 - uap) ** 3)) * (t1p * t2 + t3)) / n

    return KappaStatistics(
        sum_n=float(n),
        sum_naive=float(th1),
        sum_var=float(th1v),
        sum_kappa=float(kh),
        sum_kvar=float(khv),
        user_naive=ua,
        prod_naive=pa,
        user_kappa=kpu,
        user_kvar=kpuv,
        prod_kappa=kpp,
        prod_kvar=kppv,
    )


@dataclass
class DatasetCheckReport:
    """Structured result of :func:`check_dataset`, for callers (the GUI's
    Validate button) that need the actual per-dataset issue list rather
    than just a bool."""

    ok: bool
    issues: List[str]


def _grids_aligned(a: dict, b: dict) -> bool:
    """Do two raster profiles share the same shape, resolution/origin, and
    CRS? Ported concept, not a direct R port: R's ``isDataSetCorrect``
    relies on ``raster::stack()`` throwing a hard error first if any layer's
    extent/resolution/CRS doesn't match everything else in the stack,
    *before* its NA-ratio arithmetic ever runs. The Python port historically
    dropped that guarantee (it reads each layer independently), so two
    differently-aligned rasters could coincidentally produce a matching
    valid-cell ratio and be reported "correct". This check restores it
    explicitly — with a tolerance on the transform, because even genuinely
    co-registered real-world rasters carry sub-micrometer floating-point
    noise in their affine origin (observed ~2e-9 map units on the bundled
    sample data — negligible against a 200m pixel size, but not exactly
    zero), so exact equality on the transform is too strict.
    """
    if (a.get("height"), a.get("width")) != (b.get("height"), b.get("width")):
        return False
    if a.get("crs") != b.get("crs"):
        return False
    ta, tb = a.get("transform"), b.get("transform")
    if (ta is None) != (tb is None):
        return False
    if ta is not None and not all(
        math.isclose(x, y, rel_tol=1e-9, abs_tol=1e-6) for x, y in zip(ta, tb)
    ):
        return False
    return True


def check_dataset(
    t1_drivers: Dict[str, str],
    t2_drivers: Dict[str, str],
    t1_file: str,
    t2_file: str,
    t3_file: Optional[str] = None,
    extra_layers: Optional[Dict[str, str]] = None,
) -> DatasetCheckReport:
    """Check that every input raster is spatially aligned with, and has a
    matching valid-cell footprint to, T2.

    Port of R's ``isDataSetCorrect`` (lines 2491-2536), generalized in three
    ways: extent/resolution/CRS alignment is checked explicitly (see
    :func:`_grids_aligned`) instead of relying on ``raster::stack()`` to fail,
    ``extra_layers`` lets callers check named rasters that aren't
    drivers (e.g. the GUI's AreaOfInterest/Mask fields) through the same
    machinery, and each layer's own NA/nodata *value* is compared against
    T2's (not just the valid-cell *ratio* — two rasters can have the same
    count of NA cells while actually flagging different pixel values as NA,
    which the ratio check alone can't catch). A layer with a grid mismatch
    is reported as such and neither of the other two checks run for it (not
    meaningful across misaligned grids); an aligned layer is checked for a
    matching NA value and then for a matching valid-cell ratio, preserving
    R's exact "LESS"/"MORE" wording for the latter.

    This NA-value check matters because the pipeline applies a single
    ``na_value`` override uniformly to every raster it reads (see
    ``raster_io.read_raster``): if the rasters don't actually share one
    NA value natively, no single override can be correct for all of them —
    whichever ones it's wrong for will have their true NA cells silently
    treated as an ordinary class/value instead of being filtered out.
    """
    layers: List[tuple] = []
    for name, path in t1_drivers.items():
        layers.append((f"T1Driver.{name}", path))
    for name, path in t2_drivers.items():
        layers.append((f"T2Driver.{name}", path))
    layers.append(("T1File", t1_file))
    layers.append(("T2File", t2_file))
    if t3_file is not None:
        layers.append(("T3File", t3_file))
    for name, path in (extra_layers or {}).items():
        layers.append((name, path))

    t2_layer = read_raster(t2_file)
    t2_valid = int(t2_layer.valid_mask().sum())
    if t2_valid == 0:
        logger.error("T2 raster has no valid cells")
        return DatasetCheckReport(ok=False, issues=["T2LULC has no valid cells"])

    t2_nodata = t2_layer.nodata

    issues: List[str] = []
    for name, path in layers:
        layer = read_raster(path)
        if not _grids_aligned(layer.profile, t2_layer.profile):
            issues.append(f"{name.upper()} extent does not match T2LULC (shape/resolution/CRS)")
            continue

        nodata = layer.nodata
        same_nodata = (
            nodata is None and t2_nodata is None
        ) or (
            nodata is not None and t2_nodata is not None
            and math.isclose(nodata, t2_nodata, rel_tol=1e-9, abs_tol=1e-6)
        )
        if not same_nodata:
            issues.append(
                f"{name.upper()} NA value ({nodata}) does not match T2LULC NA value ({t2_nodata})"
            )
            continue  # a ratio comparison across differing NA values isn't meaningful either

        ratio = layer.valid_mask().sum() / t2_valid
        if ratio != 1:
            side = "LESS" if ratio < 1 else "MORE"
            issues.append(f'{name.upper()} has "{side}" Number of NA Compare to T2LULC')

    ok = not issues
    if ok:
        logger.info("No Error in Data Sets")
    else:
        for issue in issues:
            logger.warning(issue)
    return DatasetCheckReport(ok=ok, issues=issues)


def is_dataset_correct(
    t1_drivers: Dict[str, str],
    t2_drivers: Dict[str, str],
    t1_file: str,
    t2_file: str,
    t3_file: Optional[str] = None,
) -> bool:
    """Check that every input raster's valid-cell footprint matches T2's.

    Thin bool-returning wrapper around :func:`check_dataset`, kept for
    existing callers (``OpenLDM.py``'s CLI path) that only need the pass/fail
    result. Unlike R (which returns NULL on failure), this always returns a
    bool.
    """
    return check_dataset(t1_drivers, t2_drivers, t1_file, t2_file, t3_file).ok
