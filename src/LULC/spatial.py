"""Neighborhood / focal spatial operations, array-native.

Ports:

* ``ComputeNW`` / ``ParallelComputeNearByWeight`` -> :func:`compute_nearby_weights`
* the ``focal(...)`` neighborhood-density blending used inside
  ``genratePredictedMap``'s multi-step branch -> :func:`focal_density`

The Markov ``get_yearly_transition_matrix`` that previously lived here moved
to :mod:`transition` (and was re-derived from the R source's live behavior).

Faithfulness notes (Rasterise_dev_68akj.r lines 2069-2187, 2300-2321):

* ``ParallelComputeNearByWeight`` contains ``if(!is.na(wSize)){ wSize=3 }`` —
  any supplied window size is **forced to 3** before the distance clamp. This
  is reachable, output-affecting behavior (the multi-step branch always calls
  with the GUI's window size), so it is replicated here.
* Distance weights are ``as.integer(dst / min(dst[dst > 0]))``: Euclidean
  distance to the nearest cell of the class, divided by the smallest positive
  distance found anywhere on the grid, truncated to integer. Cells of the
  class itself get weight 0; every other cell gets weight >= 1.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np
from joblib import Parallel, delayed
from scipy import ndimage

from .config import PARALLEL_BACKEND, PARALLEL_JOBS, logger


def _nearby_weight_single(
    class_mask: np.ndarray,
    pixel_size_xy: tuple,
    window_size: Optional[int],
) -> np.ndarray:
    """Integer distance-to-nearest-class weights for one class (R: ComputeNW)."""
    xres, yres = pixel_size_xy
    if not class_mask.any():
        return np.zeros(class_mask.shape, dtype=np.int64)

    dst = ndimage.distance_transform_edt(~class_mask, sampling=(abs(yres), abs(xres)))

    if window_size is not None:
        # R: ParallelComputeNearByWeight forces wSize=3 before ComputeNW's clamp.
        forced = 3
        lardist = float(np.sqrt(abs(xres) * abs(yres)) * forced)
        dst = np.minimum(dst, lardist)

    positive = dst[dst > 0]
    min_positive = positive.min() if positive.size else 1.0
    return (dst / min_positive).astype(np.int64)


def compute_nearby_weights(
    class_codes: np.ndarray,
    class_ids: Sequence[int],
    pixel_size_xy: tuple,
    window_size: Optional[int] = None,
    n_jobs: int = PARALLEL_JOBS,
) -> Dict[int, np.ndarray]:
    """Per-class integer proximity weights on the full grid.

    Port of R's ``ParallelComputeNearByWeight`` (one worker per class). Returns
    ``{class_id: (rows, cols) int array}``; weight 0 marks cells currently of
    that class, larger values are farther from the nearest occurrence.
    """
    logger.info(f"Computing neighborhood weights for {len(class_ids)} classes...")
    masks = [(class_codes == cid) for cid in class_ids]
    n_jobs = min(len(class_ids), n_jobs) if n_jobs > 0 else n_jobs
    results = Parallel(n_jobs=n_jobs, backend=PARALLEL_BACKEND)(
        delayed(_nearby_weight_single)(mask, pixel_size_xy, window_size) for mask in masks
    )
    return {cid: w for cid, w in zip(class_ids, results)}


def focal_density(
    presence: np.ndarray,
    window_size: int,
    invalid_mask: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Windowed count of class presence around each cell.

    Port of the multi-step branch's ``focal(tmp, w=windowMatrix, pad=TRUE,
    padValue=0)`` followed by ``f[is.na(t2)] <- NA`` and ``f[f == 0] <- 1``
    (Rasterise_dev_68akj.r lines 2312-2315). Returns a float array where
    invalid cells are NaN and zero-neighborhood cells are lifted to 1.
    """
    kernel = np.ones((window_size, window_size))
    density = ndimage.convolve(presence.astype(float), kernel, mode="constant", cval=0.0)
    density = np.rint(density)
    density[density == 0] = 1.0
    if invalid_mask is not None:
        density[invalid_mask] = np.nan
    return density
