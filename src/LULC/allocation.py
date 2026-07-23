"""Competitive land allocation engine.

Faithful, array-native port of R's ``getAllocatedDT``
(Rasterise_dev_68akj.r lines 28-374) and ``removeAllocateGrid`` (379-387).

The four phases of the R algorithm are preserved exactly:

1. **Inertia retention** — for each class with inertia > 0, lock
   ``floor(inertia * TM[i,i])`` of its current cells to stay put. R sorts the
   candidates by suitability **ascending** and locks the first n — i.e. the
   *least* suitable current cells are guaranteed retention, leaving the most
   suitable ones in the competitive pool. Reproduced as-is.
2. **Competitive allocation** — outer loop over ``classAllocationOrder``
   (which orders the **from** classes), inner loop over the conversion-order
   priority list of target classes. Candidate cells are unallocated cells of
   the from-class; ordering depends on the demand share:
   * share > 5% of the target row: keep the top-``demand`` cells by target
     suitability (with ties past the cutoff), then allocate in order of
     **proximity to existing target-class patches** (neighbour weight
     ascending).
   * share <= 5%: the R code's final reordering step compares ids against a
     whole data.table with ``%in%`` (line 213), which matches nothing — the
     branch therefore allocates **zero cells**, leaving the demand for the
     fallback phases. Reproduced (with a log warning), per the
     preserve-live-behavior rule for faithfully porting observed R behavior.
3. **Optimum fallback** — any from-class whose remaining unallocated current
   cells exactly equal its remaining outbound demand gets all of them
   reassigned to itself (R warns "allocation cost Increasing"). Cells
   allocated in this phase are **not** removed from the candidate pools (R
   omits the removeAllocateGrid call here) — which is exactly what makes
   phase 4 able to re-assign them.
4. **Retreat pass** — re-attempts leftover demand restricted to cells already
   allocated to the from-class. Because phases 1/2 removed their allocations
   from the pools, only phase-3 allocations are still visible here, so the
   retreat pass can only re-assign those.

Tie-breaking: every R sort is data.table's stable ``order()`` over tables
initially keyed by id, so within equal weights ids ascend. All numpy sorts
below use ``np.lexsort`` with id as the secondary key to match exactly.

Returns a flat int array over all grid cells: allocated class id per cell, 0
where unallocated/NA (R returns the equivalent one-hot table with NA rows).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import numpy as np

from .config import logger


@dataclass
class AllocationResult:
    allocated: np.ndarray  # flat int array, 0 = unallocated/NA
    warnings: List[str] = field(default_factory=list)


def derive_conversion_order(transition_matrix: np.ndarray) -> np.ndarray:
    """R's conversionOrder='TP' derivation: row i is the list of target-class
    indices (0-based) sorted by transition count descending, ties by column
    ascending."""
    tm = np.asarray(transition_matrix)
    n = tm.shape[0]
    order = np.empty((n, n), dtype=int)
    cols = np.arange(n)
    for i in range(n):
        order[i] = np.lexsort((cols, -tm[i]))
    return order


def _ordered(ids: np.ndarray, weights: np.ndarray, descending: bool) -> np.ndarray:
    """ids sorted by weight (NaN last), ties by id ascending — matching R's
    stable order() on id-keyed tables."""
    w = weights.astype(float)
    nan_mask = np.isnan(w)
    if descending:
        w = np.where(nan_mask, -np.inf, w)
        order = np.lexsort((ids, -w))
    else:
        w = np.where(nan_mask, np.inf, w)
        order = np.lexsort((ids, w))
    return ids[order]


def get_allocated(
    suitability: Dict[int, np.ndarray],
    transition_matrix: np.ndarray,
    current_codes: np.ndarray,
    neighbour_weights: Dict[int, np.ndarray],
    class_ids: Sequence[int],
    pool_ids: np.ndarray,
    restrict_spatial_migration: Optional[Sequence[float]] = None,
    conversion_order: Union[str, np.ndarray] = "TP",
    class_allocation_order: Optional[Sequence[int]] = None,
) -> AllocationResult:
    """Allocate future land use (R: getAllocatedDT).

    Args:
        suitability: {class_id: (rows, cols) float suitability array}.
        transition_matrix: target cell-count matrix (from x to), 1 row/col
            per entry of ``class_ids``.
        current_codes: (rows, cols) current class-code array.
        neighbour_weights: {class_id: (rows, cols) int proximity array}
            (0 = cell currently of that class).
        class_ids: 1-based class codes, in matrix row/col order.
        pool_ids: flat indices of cells eligible for allocation (R: the
            suitability tables' id set — cells with usable driver data).
        restrict_spatial_migration: per-class inertia in [0, 1].
        conversion_order: 'TP' (derive from transition_matrix) or an
            (n, n) array whose row i lists 1-based target class ids in
            priority order.
        class_allocation_order: 1-based from-class processing order.
    """
    class_ids = list(class_ids)
    n = len(class_ids)
    id_of = {cid: k for k, cid in enumerate(class_ids)}

    new_tm = np.array(transition_matrix, dtype=float).copy()
    warnings: List[str] = []

    flat_codes = np.asarray(current_codes).ravel()
    n_cells = flat_codes.size

    suit_flat = {cid: np.asarray(s).ravel() for cid, s in suitability.items()}
    nw_flat = {cid: np.asarray(w).ravel() for cid, w in neighbour_weights.items()}

    allocated = np.zeros(n_cells, dtype=int)  # 0 = unallocated (R: NA row)
    in_pool = np.zeros(n_cells, dtype=bool)
    in_pool[pool_ids] = True

    if restrict_spatial_migration is None:
        inertia = np.zeros(n)
    else:
        inertia = np.asarray(restrict_spatial_migration, dtype=float)

    if class_allocation_order is None:
        from_order = list(range(n))
    else:
        from_order = [id_of[c] for c in class_allocation_order]

    if isinstance(conversion_order, str) and conversion_order == "TP":
        conv = derive_conversion_order(new_tm)
    else:
        conv = np.asarray(conversion_order, dtype=int)
        conv = np.vectorize(id_of.get)(conv)  # 1-based class ids -> 0-based indices

    def candidates(mask: np.ndarray) -> np.ndarray:
        return np.nonzero(mask)[0]

    # ---- Phase 1: inertia retention (R lines 114-144) -----------------------
    for k, cid in enumerate(class_ids):
        if inertia[k] <= 0:
            continue
        n_lock = int(np.floor(inertia[k] * new_tm[k, k]))
        if n_lock <= 0:
            continue
        cand = candidates(in_pool & (nw_flat[cid] == 0))
        # R sorts ascending by suitability and locks the first n.
        cand = _ordered(cand, suit_flat[cid][cand], descending=False)
        n_lock = min(n_lock, cand.size)
        lock_ids = cand[:n_lock]
        allocated[lock_ids] = cid
        in_pool[lock_ids] = False
        new_tm[k, k] -= n_lock

    # ---- Phase 2: competitive allocation (R lines 154-254) ------------------
    for from_idx in from_order:
        from_cid = class_ids[from_idx]
        for j in range(n):
            to_idx = int(conv[from_idx, j])
            to_cid = class_ids[to_idx]
            demand = new_tm[from_idx, to_idx]
            if demand <= 0:
                continue

            cand = candidates(in_pool & (allocated == 0) & (nw_flat[from_cid] == 0))
            if to_idx != from_idx:
                cand = cand[nw_flat[to_cid][cand] != 0]
                n_cand = cand.size
                if n_cand > demand:
                    row_total = new_tm[to_idx].sum()
                    share = demand / row_total if row_total > 0 else np.inf
                    if share > 0.05:
                        by_suit = _ordered(cand, suit_flat[to_cid][cand], descending=True)
                        cutoff = suit_flat[to_cid][by_suit[int(demand) - 1]]
                        keep = cand[suit_flat[to_cid][cand] >= cutoff]
                        chosen_order = _ordered(keep, nw_flat[to_cid][keep].astype(float), descending=False)
                    else:
                        # R's small-share branch ends by matching ids against a
                        # data.table with %in%, which never matches: nothing is
                        # allocated here (see module docstring).
                        msg = (
                            f"Small-share transition {from_cid}->{to_cid} "
                            f"(share <= 5%): R allocates nothing here; deferring "
                            f"{int(demand)} cells to fallback phases"
                        )
                        logger.warning(msg)
                        warnings.append(msg)
                        chosen_order = np.array([], dtype=int)
                else:
                    if n_cand < demand:
                        msg = f"Not Enough Probable Grid from Class {from_cid} for to Class {to_cid}"
                        logger.warning(msg)
                        warnings.append(msg)
                    # count <= demand: all candidates are taken; R's ordering
                    # differences here don't change the selected set.
                    chosen_order = _ordered(cand, suit_flat[to_cid][cand], descending=True)
            else:
                chosen_order = _ordered(cand, suit_flat[from_cid][cand], descending=True)

            n_take = min(int(demand), chosen_order.size)
            if n_take > 0:
                take = chosen_order[:n_take]
                allocated[take] = to_cid
                in_pool[take] = False
                new_tm[from_idx, to_idx] -= n_take

    # ---- Phase 3: optimum fallback (R lines 256-274) -------------------------
    if new_tm.sum() != 0:
        row_remaining = new_tm.sum(axis=1)
        for from_idx in range(n):
            from_cid = class_ids[from_idx]
            cand = candidates(in_pool & (allocated == 0) & (nw_flat[from_cid] == 0))
            if row_remaining[from_idx] > 0 and cand.size == row_remaining[from_idx]:
                allocated[cand] = from_cid
                # R does NOT remove these from the pools — phase 4 depends on it.
                msg = "allocation cost Increasing"
                logger.warning(msg)
                warnings.append(msg)

        # ---- Phase 4: retreat pass (R lines 282-367) --------------------------
        for from_idx in from_order:
            from_cid = class_ids[from_idx]
            for j in range(n):
                to_idx = j
                to_cid = class_ids[to_idx]
                demand = new_tm[from_idx, to_idx]
                if demand <= 0:
                    continue

                assigned_here = candidates(in_pool & (allocated == from_cid))
                if to_idx != from_idx:
                    cand = assigned_here[nw_flat[to_cid][assigned_here] != 0]
                    fromto = assigned_here[nw_flat[from_cid][assigned_here] == 0]
                    if fromto.size > demand and cand.size > demand:
                        by_suit = _ordered(cand, suit_flat[to_cid][cand], descending=True)
                        cutoff = suit_flat[to_cid][by_suit[int(demand) - 1]]
                        keep = cand[suit_flat[to_cid][cand] >= cutoff]
                        chosen_order = _ordered(keep, nw_flat[to_cid][keep].astype(float), descending=False)
                    else:
                        msg = f"Increasing Cost no help {from_cid} for to Class {to_cid}"
                        logger.warning(msg)
                        warnings.append(msg)
                        chosen_order = _ordered(cand, suit_flat[to_cid][cand], descending=True)
                else:
                    cand = assigned_here[nw_flat[from_cid][assigned_here] == 0]
                    chosen_order = _ordered(cand, suit_flat[from_cid][cand], descending=True)

                n_take = min(int(demand), chosen_order.size)
                if n_take > 0:
                    take = chosen_order[:n_take]
                    allocated[take] = to_cid
                    in_pool[take] = False
                    new_tm[from_idx, to_idx] -= n_take

    leftover = new_tm.sum()
    if leftover != 0:
        msg = f"Allocation finished with {int(leftover)} unmet transition-demand cells"
        logger.warning(msg)
        warnings.append(msg)

    return AllocationResult(allocated=allocated, warnings=warnings)
