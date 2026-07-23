"""Shared configuration, logging, and typed run-configuration objects.

Large positional/keyword argument lists (mirroring the R
``genratePredictedMap`` signature) are represented as typed dataclasses
internally. ``LULCAlgorithms.generate_predicted_map`` keeps the flat,
R-derived keyword-argument surface as a thin backward-compatible wrapper
around :class:`RunConfig`, since the GUI (a future migration slice) is
expected to keep using that calling convention.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

# Global defaults (kept for backward compatibility within the package).
NA_VALUE = 128
PARALLEL_JOBS = -1  # joblib convention: -1 = use all available cores


def _safe_parallel_backend() -> Optional[str]:
    """joblib's default Parallel backend (loky) spawns real OS worker
    *processes*, launched by re-invoking sys.executable -- fine for a
    normal `python ...` process, but inside a QGIS plugin sys.executable
    is QGIS's own application binary (confirmed for real: sys.executable
    there is literally ".../QGIS.app/Contents/MacOS/QGIS"), so loky ends
    up trying to spawn a fresh copy of the whole GUI application as a
    "worker" -- which crashes rather than running a task and returning a
    result.

    "threading" sidesteps needing to spawn any subprocess at all (no new
    process, just threads inside this already-running interpreter); the
    numpy/scikit-learn calls each parallelized task actually spends its
    time in release the GIL during the heavy lifting, so this still
    parallelizes meaningfully rather than degrading to fully serial.

    Detected generically (by whether sys.executable *looks* like a
    Python interpreter), not via any QGIS/Qt-specific check -- this
    module has to stay Qt-free and framework-agnostic; any
    host application that similarly replaces sys.executable with its own
    binary hits the same fix, not just QGIS specifically."""
    exe_name = os.path.basename(sys.executable or "").lower()
    if "python" in exe_name:
        return None  # joblib's own default (loky) is safe here
    return "threading"


PARALLEL_BACKEND = _safe_parallel_backend()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("OpenLDM")


@dataclass
class ClassConfig:
    """Per-LULC-class configuration.

    ``class_id`` is the 1-based pixel value of this class in the categorical
    LULC rasters, matching the R backend's convention throughout.
    """

    name: str
    class_id: int
    model_type: str = "logistic"
    model_formula: Optional[Sequence[str]] = None  # explicit driver subset; None = use all drivers
    inertia: float = 0.0  # R: restrictSpatialMigration[i], in [0, 1]
    demand: Optional[float] = None  # R: mydemand[i], target pixel count


@dataclass
class NeighbourhoodConfig:
    """R: neighbour = list(windowSize, NoOfsteps, writeStepFile)."""

    window_size: Optional[int] = None
    steps: int = 1
    write_step_output: bool = False


@dataclass
class RunConfig:
    """Full parameterization of a single prediction/allocation run.

    Mirrors R's ``genratePredictedMap`` parameters, grouped by class via
    :class:`ClassConfig` instead of parallel arrays.
    """

    t1_file: str
    t2_file: str
    classes: List[ClassConfig]
    t1_drivers: Dict[str, str]
    t2_drivers: Dict[str, str]
    output_file: str
    na_value: int = NA_VALUE
    conversion_order: Union[str, Sequence[int]] = "TP"
    class_allocation_order: Optional[Sequence[int]] = None
    neighbourhood: Optional[NeighbourhoodConfig] = None
    method: str = "NotIncludeCurrentClass"
    mask_file: Optional[str] = None
    aoi_file: Optional[str] = None
    suitability_file_directory: Optional[str] = None
    parallel_jobs: int = PARALLEL_JOBS

    @property
    def class_names(self) -> List[str]:
        return [c.name for c in self.classes]

    @property
    def model_types(self) -> List[str]:
        return [c.model_type for c in self.classes]

    @property
    def model_formulas(self) -> List[Optional[Sequence[str]]]:
        return [c.model_formula for c in self.classes]

    @property
    def restrict_spatial_migration(self) -> List[float]:
        return [c.inertia for c in self.classes]

    @property
    def demand(self) -> Optional[List[float]]:
        if all(c.demand is None for c in self.classes):
            return None
        return [c.demand if c.demand is not None else 0.0 for c in self.classes]

    @property
    def class_allocation_order_resolved(self) -> List[int]:
        if self.class_allocation_order is not None:
            return list(self.class_allocation_order)
        return [c.class_id for c in self.classes]


@dataclass
class PipelineResult:
    """Return value of :func:`LULCAlgorithms.run_pipeline`."""

    output_file: str
    transition_matrix: Any  # np.ndarray; loosely typed to avoid importing numpy here
    yearly_transition_matrix: Any
    model_summary: Dict[str, str] = field(default_factory=dict)
    step_files: List[str] = field(default_factory=list)
    suitability_files: Dict[str, str] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
