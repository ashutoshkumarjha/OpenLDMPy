"""Transition-matrix (Markov chain) logic.

Ports, faithfully to observed R behavior:

* ``createTM``                        -> :func:`build_transition_matrix`
* ``removeTruncitionErrorFromMatrix`` -> :func:`redistribute_truncation_error`
* ``getNewTM`` / ``getNewTMwithDemand`` -> :func:`get_new_transition_matrix`
* ``getYearlyMatrix``                 -> :func:`get_yearly_matrix`

R's ``minMAB``/``maxMAB`` are hand-rolled equivalents of ``np.minimum`` /
``np.maximum`` and are used directly where needed instead of being ported.

Faithfulness notes (verified against Rasterise_dev_68akj.r lines 1120-1326):

* ``getYearlyMatrix`` computed an elaborate generator-matrix series expansion
  and then **discarded it** — a since-fixed R bug where the ``expm`` result
  was immediately overwritten by a simple one-step Markov projection,
  silently ignoring ``Steps``/``foryear``. Now that the R source is
  fixed, :func:`get_yearly_matrix` implements the *intended* continuous-time
  Markov interpolation via ``scipy.linalg.logm``/``expm`` rather than
  replicating R's hand-rolled truncated-series approximation of the same
  quantity — cleaner and numerically more robust, consistent with this
  codebase's general preference for library primitives over R's internals
  where byte-exact replication isn't the goal (see ``modeling.py``).
* ``removeTruncitionErrorFromMatrix`` picks its adjustment cell with R's
  ``which.max``, which flattens matrices in column-major order — replicated
  here via ``order='F'`` flattening so tie-breaks match R exactly.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import scipy.linalg

from .config import logger


def build_transition_matrix(
    t1_codes: np.ndarray,
    t2_codes: np.ndarray,
    class_ids: Sequence[int],
    valid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Count matrix M[i, j] = number of cells with T1 class ``class_ids[i]``
    and T2 class ``class_ids[j]``.

    Port of R's ``createTM`` (lines 1181-1222). R encodes each cell's class
    pair as ``t1*(N+1)+t2`` and tabulates; cells that are NA in either raster
    produce keys that match no (i, j) pair and are therefore excluded — the
    same effect as counting only cells whose codes are both in ``class_ids``.
    """
    ids = np.asarray(sorted(class_ids))
    n_classes = ids.size
    t1 = np.asarray(t1_codes).ravel()
    t2 = np.asarray(t2_codes).ravel()
    if valid_mask is not None:
        keep = np.asarray(valid_mask).ravel()
        t1, t2 = t1[keep], t2[keep]

    finite = np.isfinite(t1) & np.isfinite(t2)
    t1, t2 = t1[finite], t2[finite]
    i1 = np.searchsorted(ids, t1)
    i2 = np.searchsorted(ids, t2)
    ok = (
        (i1 < n_classes)
        & (i2 < n_classes)
        & (ids[np.minimum(i1, n_classes - 1)] == t1)
        & (ids[np.minimum(i2, n_classes - 1)] == t2)
    )
    i1, i2 = i1[ok].astype(np.int64), i2[ok].astype(np.int64)

    flat = i1 * n_classes + i2
    counts = np.bincount(flat, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes).astype(np.int64)


def _which_max_colmajor(m: np.ndarray) -> tuple:
    """R's ``which.max`` on a matrix: first maximum in column-major order."""
    idx = int(np.argmax(m.flatten(order="F")))
    return idx % m.shape[0], idx // m.shape[0]


def redistribute_truncation_error(
    reference_vector: np.ndarray,
    m: np.ndarray,
    by: str,
) -> np.ndarray:
    """Nudge one cell per row/col so marginal sums match ``reference_vector``.

    Faithful port of R's ``removeTruncitionErrorFromMatrix`` (lines 1224-1248),
    including its quirks: the row branch's sign test inspects **column** ``i``
    (as the R source does), and the final global adjustment lands on the
    column-major-first maximum cell.
    """
    m = np.array(m, dtype=float)
    ref = np.asarray(reference_vector, dtype=float)
    n = m.shape[0]

    if by == "row":
        diff = ref - m.sum(axis=1)
        for i in range(n):
            if np.any(m[:, i] >= 0):
                loc = int(np.argmax(m[i, :]))
            else:
                loc = int(np.argmin(m[i, :]))
            if diff[i] == ref[i]:
                loc = i
            m[i, loc] += diff[i]
    elif by == "col":
        diff = ref - m.sum(axis=0)
        for i in range(n):
            if np.any(m[:, i] >= 0):
                loc = int(np.argmax(m[:, i]))
            else:
                loc = int(np.argmin(m[:, i]))
            if diff[i] == ref[i]:
                loc = i
            m[loc, i] += diff[i]
    else:
        raise ValueError(f"by must be 'row' or 'col', got {by!r}")

    r, c = _which_max_colmajor(m)
    m[r, c] += ref.sum() - m.sum()
    return m


def _markov_projection(tm: np.ndarray) -> np.ndarray:
    """R: round(colSums(TM) * TM / rowSums(TM)).

    Element (i, j) = colSums(TM)[i] * TM[i, j] / rowSums(TM)[i] — i.e. row i
    rescaled so its total equals the T2 count of class i+1: one further
    Markov step applied to the T2 class distribution.
    """
    tm = np.asarray(tm, dtype=float)
    row_sums = tm.sum(axis=1)
    col_sums = tm.sum(axis=0)
    safe_rows = np.where(row_sums == 0, 1.0, row_sums)
    projected = (col_sums[:, np.newaxis] * tm) / safe_rows[:, np.newaxis]
    return np.round(projected)


def get_new_transition_matrix(
    tm: np.ndarray,
    demand: Optional[Sequence[float]] = None,
    spatial_migration_restriction: Optional[Sequence[float]] = None,
    max_ipf_iterations: int = 1000,
) -> np.ndarray:
    """Target future transition matrix.

    Port of R's ``getNewTM`` (lines 1262-1303) and ``getNewTMwithDemand``
    (lines 1305-1326). Without demand: one-step Markov projection. With
    demand: IPF-style alternation forcing row totals to the T2 distribution
    and column totals to ``demand``.

    ``max_ipf_iterations`` is a safety cap absent in R (whose ``while`` loop
    can spin forever on infeasible demand); hitting it logs a warning and
    returns the best iterate, rather than hanging.
    """
    tm = np.asarray(tm, dtype=float)
    n = tm.shape[0]
    col_sums = tm.sum(axis=0)

    if demand is None:
        future = _markov_projection(tm)
    else:
        future = _get_new_tm_with_demand(tm, np.asarray(demand, dtype=float), max_ipf_iterations)

    future = redistribute_truncation_error(col_sums, future, "row")

    diff = col_sums - future.sum(axis=1)
    for i in range(n):
        future[i, i] += diff[i]

    return np.round(future)


def _get_new_tm_with_demand(tm: np.ndarray, demand: np.ndarray, max_iterations: int) -> np.ndarray:
    col_sums = tm.sum(axis=0)
    fem = _markov_projection(tm)

    iteration = 0
    while np.any(demand - fem.sum(axis=0) != 0):
        if iteration >= max_iterations:
            logger.warning(
                "Demand-constrained transition matrix did not converge after "
                f"{max_iterations} IPF iterations; residual "
                f"{demand - fem.sum(axis=0)}. Check that demand sums to the "
                "total valid cell count."
            )
            break
        row_sums = fem.sum(axis=1)
        safe_rows = np.where(row_sums == 0, 1.0, row_sums)
        fem = np.round(fem / safe_rows[:, np.newaxis] * col_sums[:, np.newaxis])
        fem = redistribute_truncation_error(col_sums, fem, "row")
        fem_col_sums = fem.sum(axis=0)
        safe_cols = np.where(fem_col_sums == 0, 1.0, fem_col_sums)
        fem = np.round(fem / safe_cols[np.newaxis, :] * demand[np.newaxis, :])
        fem = redistribute_truncation_error(demand, fem, "col")
        iteration += 1

    return fem


def get_yearly_matrix(fcm: np.ndarray, steps: int = 1, for_year: int = 1) -> np.ndarray:
    """Per-step transition matrix for the multi-step simulation loop.

    Port of R's ``getYearlyMatrix``'s *intended* behavior (its own R-side
    bug is now fixed alongside this change): estimates a
    continuous-time-Markov generator matrix Q from the one-step
    row-stochastic matrix via the matrix logarithm, scales it by
    ``1/steps`` and by ``for_year``, and exponentiates back to a
    transition-probability matrix — so an intermediate simulation step
    (``for_year`` between 1 and ``steps``) gets a genuinely interpolated
    matrix instead of the full-period one repeated unchanged.
    """
    fcm = np.asarray(fcm, dtype=float)
    row_sums = fcm.sum(axis=1)
    safe_rows = np.where(row_sums == 0, 1.0, row_sums)
    fpm = fcm / safe_rows[:, np.newaxis]  # row-stochastic one-step matrix
    q = scipy.linalg.logm(fpm).real / steps
    p = scipy.linalg.expm(q * for_year)
    projected = np.round(p * row_sums[:, np.newaxis])
    return redistribute_truncation_error(row_sums, projected, "row")
