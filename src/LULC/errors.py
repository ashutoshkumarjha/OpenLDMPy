"""Custom exception hierarchy (custom exceptions instead of letting
the application terminate on an uncaught error).

Raised by processing-layer code for user-actionable failure modes; anything
else (programming errors, I/O failures) propagates as whatever built-in
exception it naturally raises. Callers (CLI, GUI) catch `PipelineError` to
show a clean message instead of a traceback, and can still catch `Exception`
broadly as a fallback.
"""

from __future__ import annotations


class PipelineError(Exception):
    """Base class for user-actionable failures in the LULC pipeline."""


class DatasetValidationError(PipelineError):
    """Raised when input rasters are inconsistent (misaligned grids, class
    codes outside the configured set, mismatched NA footprints via
    is_dataset_correct)."""


class AllocationError(PipelineError):
    """Raised when the competitive allocation engine cannot proceed at all
    (e.g. no eligible cells for a class that has a nonzero transition
    matrix). Partial/unmet demand is reported via warnings, not this
    exception — see allocation.AllocationResult.warnings."""


class RasterizeError(PipelineError):
    """Raised by rasterize.rasterise() for user-actionable shapefile schema
    problems: missing class_field/poly_id_field column, an unsupported
    `option`, or an empty/geometry-less shapefile."""
