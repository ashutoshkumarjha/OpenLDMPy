"""Synthetic, hand-computed tests for allocation.py (no R oracle — R's
allocation loop has no isolated fixture path)."""

import numpy as np
import pytest

from LULC import allocation


def _grid(codes):
    return np.array(codes, dtype=float)


def test_demand_is_satisfied_when_feasible():
    """3x3 grid, classes 1 and 2. Transition matrix asks for 2 cells of
    class 1 to become class 2; suitability strongly favors two specific
    cells for conversion, and (deliberately) disfavors those same two cells
    for self-persistence, so the self-transition phase (processed first,
    since it is the larger count per the TP-derived conversion order) does
    not consume them first."""
    current = _grid([[1, 1, 1], [1, 1, 1], [2, 2, 2]])
    n = current.size
    pool_ids = np.arange(n)

    suit_1 = np.array([[0.1, 0.1, 0.9], [0.9, 0.9, 0.9], [0, 0, 0]])  # self-persistence preference
    suit_2 = np.zeros_like(current)
    suit_2.flat[[0, 1]] = [0.9, 0.8]  # top two candidates for -> class 2
    suitability = {1: suit_1, 2: suit_2}

    nw = {1: np.where(current == 1, 0, 1), 2: np.where(current == 2, 0, 1)}

    # class1->class1:4, class1->class2:2 (totals the 6 available class-1
    # cells exactly; R processes the larger self-transition first per the
    # TP-derived conversion order, so an infeasible total here would starve
    # the class1->class2 transition before it runs).
    tm = np.array([[4.0, 2.0], [0.0, 3.0]])

    result = allocation.get_allocated(
        suitability=suitability,
        transition_matrix=tm,
        current_codes=current,
        neighbour_weights=nw,
        class_ids=[1, 2],
        pool_ids=pool_ids,
    )
    allocated = result.allocated.reshape(current.shape)
    assert (allocated.flat[[0, 1]] == 2).all()
    assert (allocated == 2).sum() == 5  # 3 original + 2 newly converted


def test_inertia_locks_least_suitable_cells_first():
    """R sorts inertia candidates ascending and locks the first n — i.e. the
    LEAST suitable current cells are retained, not the most suitable."""
    current = _grid([[1, 1, 1, 1]])
    n = current.size
    pool_ids = np.arange(n)

    suit_1 = np.array([[0.1, 0.9, 0.5, 0.3]])
    suitability = {1: suit_1}
    nw = {1: np.zeros_like(current)}  # all cells currently class 1
    tm = np.array([[4.0]])

    result = allocation.get_allocated(
        suitability=suitability,
        transition_matrix=tm,
        current_codes=current,
        neighbour_weights=nw,
        class_ids=[1],
        pool_ids=pool_ids,
        restrict_spatial_migration=[0.5],  # lock floor(0.5*4)=2 cells
    )
    allocated = result.allocated.reshape(current.shape)
    # Least suitable (0.1, 0.3 -> indices 0, 3) locked first.
    assert allocated[0, 0] == 1 and allocated[0, 3] == 1


def test_infeasible_demand_leaves_cells_unallocated_with_warning():
    """R's getAllocatedDT has no final "revert to original class" step —
    confirmed absent from Rasterise_dev_68akj.r (grep for
    "outputAllocation[is.na]" finds nothing). Cells the algorithm can't
    place stay unallocated (id 0), which ConvertMultDimDTToMapUsingFile
    later turns into NA/nodata in the output map — they do NOT revert to
    their pre-transition class."""
    current = _grid([[1, 1]])
    pool_ids = np.array([], dtype=int)  # nothing eligible
    suitability = {1: np.zeros_like(current), 2: np.zeros_like(current)}
    nw = {1: np.zeros_like(current), 2: np.ones_like(current)}
    tm = np.array([[0.0, 2.0], [0.0, 0.0]])

    result = allocation.get_allocated(
        suitability=suitability,
        transition_matrix=tm,
        current_codes=current,
        neighbour_weights=nw,
        class_ids=[1, 2],
        pool_ids=pool_ids,
    )
    allocated = result.allocated.reshape(current.shape)
    assert (allocated == 0).all()  # unallocated, not reverted to class 1
    assert result.warnings  # unmet demand should be reported


def test_derive_conversion_order_prioritizes_larger_counts():
    tm = np.array([[5.0, 10.0, 1.0], [0, 0, 0], [0, 0, 0]])
    order = allocation.derive_conversion_order(tm)
    # Row 0: class-index 1 (count 10) should be preferred over 0 (5) over 2 (1).
    assert list(order[0]) == [1, 0, 2]


def test_no_negative_transition_matrix_entries_after_allocation():
    current = _grid([[1, 2, 1, 2]])
    pool_ids = np.arange(4)
    suitability = {1: np.random.RandomState(0).rand(1, 4), 2: np.random.RandomState(1).rand(1, 4)}
    nw = {1: np.where(current == 1, 0, 1), 2: np.where(current == 2, 0, 1)}
    tm = np.array([[1.0, 1.0], [1.0, 1.0]])

    result = allocation.get_allocated(
        suitability=suitability,
        transition_matrix=tm,
        current_codes=current,
        neighbour_weights=nw,
        class_ids=[1, 2],
        pool_ids=pool_ids,
    )
    assert result.allocated.shape == (4,)
    assert set(np.unique(result.allocated)).issubset({0, 1, 2})
