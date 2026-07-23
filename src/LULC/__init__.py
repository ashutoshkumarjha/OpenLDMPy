"""OpenLDM LULC processing package: pure-Python port of Rasterise_dev_68akj.r.

See docs/current_architecture.md and docs/function_inventory.md
for the mapping between this package and the original R backend.
"""

from . import accuracy, allocation, errors, masking, modeling, pontius, raster_io, spatial, transition
from .config import ClassConfig, NeighbourhoodConfig, PipelineResult, RunConfig, logger
from .errors import AllocationError, DatasetValidationError, PipelineError
from .LULCAlgorithms import (
    DatasetCheckReport,
    check_dataset,
    generate_predicted_map,
    get_class_codes,
    get_kappa_summary,
    get_model_fit_summary,
    is_dataset_correct,
    kappa_agreement_index,
    run_pipeline,
)
from .raster_io import RasterLayer

__all__ = [
    "accuracy",
    "allocation",
    "errors",
    "masking",
    "modeling",
    "pontius",
    "raster_io",
    "spatial",
    "transition",
    "ClassConfig",
    "NeighbourhoodConfig",
    "PipelineResult",
    "RunConfig",
    "logger",
    "PipelineError",
    "DatasetValidationError",
    "AllocationError",
    "RasterLayer",
    "DatasetCheckReport",
    "check_dataset",
    "generate_predicted_map",
    "get_class_codes",
    "get_kappa_summary",
    "get_model_fit_summary",
    "is_dataset_correct",
    "kappa_agreement_index",
    "run_pipeline",
]
