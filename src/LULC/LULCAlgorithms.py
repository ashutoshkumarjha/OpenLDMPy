"""Facade orchestrating the full LULC prediction pipeline.

This is the single entry point the CLI (``OpenLDM.py``) and, in the next
migration slice, the Qt GUI call into. It ports R's ``genratePredictedMap``
(Rasterise_dev_68akj.r lines 2190-2348) — both the single-shot and multi-step
neighborhood branches — plus ``getKappaSummary``, ``getModelFitSummary``,
``getClassNumber`` and ``isDataSetCorrect``.

Two calling conventions are provided:

* :func:`run_pipeline` takes a typed :class:`config.RunConfig` — preferred
  for new code.
* :func:`generate_predicted_map` keeps the flat R-derived keyword-argument
  surface so existing callers (the GUI's ``runRCommand`` and ``OpenLDM.py``)
  keep working unchanged.
"""

from __future__ import annotations

import os
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import rasterio as rio

from . import accuracy, allocation, cartography, masking, modeling, pontius, raster_io, rasterize, spatial, transition
from .config import ClassConfig, NeighbourhoodConfig, PipelineResult, RunConfig, logger
from .errors import DatasetValidationError

# Re-exported for callers that used these from the old facade.
is_dataset_correct = accuracy.is_dataset_correct
check_dataset = accuracy.check_dataset
DatasetCheckReport = accuracy.DatasetCheckReport


def get_class_codes(file_path: str, na_value: Optional[float] = None) -> List[int]:
    """Sorted class codes present in a categorical raster (R: getClassNumber)."""
    layer = raster_io.read_categorical_raster(file_path, na_value=na_value)
    codes = layer.single_band[layer.valid_mask()]
    return [int(c) for c in np.unique(codes)]


def rasterize_shapefile(
    shp_file: str,
    output_file: str,
    class_field: str = "IGBP_CODE",
    grid_size: float = 1000.0,
    option: str = "fraction",
    poly_id_field: Optional[str] = None,
) -> str:
    """Port of R's rasterise(): class-coded polygon shapefile -> a
    multi-band (one band per class) class-weight GeoTIFF at output_file.
    Returns output_file."""
    layer = rasterize.rasterise(
        shp_file, class_field=class_field, grid_size=grid_size,
        option=option, poly_id_field=poly_id_field,
    )
    raster_io.write_layer(layer, output_file)
    return output_file


def _pixel_size(profile: dict) -> tuple:
    t = profile["transform"]
    return (t.a, t.e)


def _write_per_class_maps(
    grids: Dict[str, np.ndarray],
    reference: raster_io.RasterLayer,
    directory: str,
    suffix: str,
) -> Dict[str, str]:
    """R: createSuitabilityMap / createNeighbourMap — one GeoTIFF per class,
    named <directory>/<className><suffix>.tif."""
    written = {}
    profile = reference.profile.copy()
    nodata = profile.get("nodata")
    for name, grid in grids.items():
        path = os.path.join(directory, f"{name}{suffix}.tif")
        arr = np.asarray(grid, dtype=float)
        if nodata is not None:
            arr = np.where(np.isnan(arr), nodata, arr)
        out_profile = profile.copy()
        out_profile.update(dtype="float32")
        raster_io.write_categorical_raster(arr, out_profile, path, nodata=nodata)
        written[name] = path
    return written


def _allocated_to_map(allocated_flat: np.ndarray, reference: raster_io.RasterLayer) -> np.ndarray:
    """R: ConvertMultDimDTToMapUsingFile — unallocated cells (0) become nodata."""
    nodata = reference.nodata
    out = allocated_flat.reshape(reference.shape).astype(float)
    out[out == 0] = nodata
    return out


def run_pipeline(
    cfg: RunConfig, on_progress: Optional[Callable[[int, str], None]] = None
) -> PipelineResult:
    """Execute the full prediction/allocation pipeline for one scenario.

    ``on_progress(percent, stage_label)``, if given, is called at this
    function's own major stage boundaries (not threaded any deeper into
    modeling/allocation/transition internals — see gui/progress_bridge.py
    and gui/log_bridge.py, which forwards fine-grained log text
    instead). Gives the GUI a real percentage instead of just log lines.
    """
    report = on_progress or (lambda _pct, _label: None)
    logger.info("--- Starting LULC Prediction (pure-Python backend) ---")
    warnings: List[str] = []

    # 1. Load inputs; apply AOI/mask exactly where R does (dT1/dT2/drivers —
    #    NOT the neighborhood weights, which R computes from the raw T2 file).
    report(5, "Loading rasters")
    t1 = raster_io.read_categorical_raster(cfg.t1_file, na_value=cfg.na_value)
    t2 = raster_io.read_categorical_raster(cfg.t2_file, na_value=cfg.na_value)
    t1 = masking.apply_mask_and_aoi(t1, cfg.mask_file, cfg.aoi_file)
    t2 = masking.apply_mask_and_aoi(t2, cfg.mask_file, cfg.aoi_file)

    drivers_t1 = masking.apply_mask_and_aoi(
        raster_io.read_driver_stack(cfg.t1_drivers), cfg.mask_file, cfg.aoi_file
    )
    drivers_t2 = masking.apply_mask_and_aoi(
        raster_io.read_driver_stack(cfg.t2_drivers), cfg.mask_file, cfg.aoi_file
    )

    class_ids = [c.class_id for c in cfg.classes]
    class_names = cfg.class_names
    name_by_id = dict(zip(class_ids, class_names))

    t1_codes = np.where(t1.valid_mask(), t1.single_band, np.nan)
    t2_codes = np.where(t2.valid_mask(), t2.single_band, np.nan)

    present = set(np.unique(t1_codes[np.isfinite(t1_codes)]).astype(int))
    if not present.issubset(set(class_ids)):
        raise DatasetValidationError(
            f"T1 raster contains class codes {sorted(present - set(class_ids))} "
            f"not covered by the configured classes {class_ids}"
        )

    # 2. Fit per-class models on T1 drivers vs T1 classes.
    report(15, "Fitting per-class models")
    models = modeling.fit_models_separately(
        t1_codes=t1_codes,
        driver_stack_t1=np.where(drivers_t1.valid_mask(), drivers_t1.array, np.nan),
        driver_names=drivers_t1.band_names,
        class_ids=class_ids,
        class_names=class_names,
        model_types=cfg.model_types,
        model_formulas=[
            f[0] if isinstance(f, (list, tuple)) and len(f) == 1 else f
            for f in cfg.model_formulas
        ],
        method=cfg.method,
    )
    model_summary = modeling.get_model_summary(models)
    for text in model_summary.values():
        logger.info("\n" + text)

    # 3. Observed transition matrix and target future matrix.
    report(30, "Building transition matrices")
    tm = transition.build_transition_matrix(t1_codes, t2_codes, class_ids)
    new_tm = transition.get_new_transition_matrix(
        tm, demand=cfg.demand, spatial_migration_restriction=cfg.restrict_spatial_migration
    )

    # 4. Suitability from T2 drivers (see modeling module docstring).
    report(45, "Predicting suitability")
    suit_by_name = modeling.construct_suitability(
        models,
        driver_stack=np.where(drivers_t2.valid_mask(), drivers_t2.array, np.nan),
        driver_names=drivers_t2.band_names,
        n_jobs=cfg.parallel_jobs,
    )
    suit = {cid: suit_by_name[name_by_id[cid]] for cid in class_ids}

    suitability_files: Dict[str, str] = {}
    if cfg.suitability_file_directory:
        suitability_files = _write_per_class_maps(
            suit_by_name, t2, cfg.suitability_file_directory, "SM"
        )

    # 5. Neighborhood weights — R computes these from the RAW (unmasked) T2
    #    file with only the NA value applied.
    report(55, "Computing neighborhood weights")
    t2_raw = raster_io.read_categorical_raster(cfg.t2_file, na_value=cfg.na_value)
    t2_raw_codes = np.where(t2_raw.valid_mask(), t2_raw.single_band, np.nan)
    pixel_size = _pixel_size(t2.profile)

    nearby = spatial.compute_nearby_weights(
        t2_raw_codes, class_ids, pixel_size, window_size=None, n_jobs=cfg.parallel_jobs
    )
    if cfg.suitability_file_directory:
        _write_per_class_maps(
            {name_by_id[cid]: w.astype(float) for cid, w in nearby.items()},
            t2,
            cfg.suitability_file_directory,
            "NW",
        )

    # Allocation pool: cells with usable suitability AND valid on the T2 map
    # (intersection of R's suitability-table and neighbour-table id sets).
    any_suit = np.isfinite(next(iter(suit.values())))
    pool_ids = np.nonzero((any_suit & np.isfinite(t2_raw_codes)).ravel())[0]

    step_files: List[str] = []

    if cfg.neighbourhood is None:
        # ---- Branch A: single-shot allocation (R lines 2253-2266) ----------
        report(70, "Allocating")
        result = allocation.get_allocated(
            suitability=suit,
            transition_matrix=new_tm,
            current_codes=t2_codes,
            neighbour_weights=nearby,
            class_ids=class_ids,
            pool_ids=pool_ids,
            restrict_spatial_migration=cfg.restrict_spatial_migration,
            conversion_order=cfg.conversion_order,
            class_allocation_order=cfg.class_allocation_order_resolved,
        )
        warnings.extend(result.warnings)
        final_map = _allocated_to_map(result.allocated, t2)
    else:
        # ---- Branch B: multi-step neighborhood simulation (R lines 2267-2341)
        nb = cfg.neighbourhood
        window = int(nb.window_size or 1)
        steps = int(nb.steps or 1)

        nearby = spatial.compute_nearby_weights(
            t2_raw_codes, class_ids, pixel_size, window_size=window, n_jobs=cfg.parallel_jobs
        )
        if cfg.suitability_file_directory:
            _write_per_class_maps(
                {name_by_id[cid]: w.astype(float) for cid, w in nearby.items()},
                t2,
                cfg.suitability_file_directory,
                "NW",
            )

        t2_work = t2_raw_codes.copy()
        invalid = ~np.isfinite(t2_raw_codes)
        weighted_suit: Dict[int, np.ndarray] = suit

        final_map = None
        for step in range(1, steps + 1):
            report(60 + int(35 * step / steps), f"Allocating step {step}/{steps}")
            yearly = transition.get_yearly_matrix(new_tm, steps, step)
            weighted_suit = {}
            for cid in class_ids:
                presence = (t2_work == cid).astype(float)
                if window != 1:
                    density = spatial.focal_density(presence, window, invalid_mask=invalid)
                    weighted_suit[cid] = suit[cid] * density
                else:
                    weighted_suit[cid] = presence

            result = allocation.get_allocated(
                suitability=weighted_suit,
                transition_matrix=yearly,
                current_codes=t2_codes,
                neighbour_weights=nearby,
                class_ids=class_ids,
                pool_ids=pool_ids,
                restrict_spatial_migration=cfg.restrict_spatial_migration,
                conversion_order=cfg.conversion_order,
                class_allocation_order=cfg.class_allocation_order_resolved,
            )
            warnings.extend(result.warnings)
            final_map = _allocated_to_map(result.allocated, t2)

            if nb.write_step_output:
                step_path = f"{cfg.output_file[:-4]}-Step{step}.tif"
                raster_io.write_categorical_raster(
                    final_map, t2.profile, step_path, nodata=t2.nodata
                )
                step_files.append(step_path)

            t2_work = np.where(final_map == t2.nodata, np.nan, final_map)

        # Final reconciliation against the whole-period matrix (R line 2340).
        result = allocation.get_allocated(
            suitability=weighted_suit,
            transition_matrix=new_tm,
            current_codes=t2_codes,
            neighbour_weights=nearby,
            class_ids=class_ids,
            pool_ids=pool_ids,
            restrict_spatial_migration=cfg.restrict_spatial_migration,
            conversion_order=cfg.conversion_order,
            class_allocation_order=cfg.class_allocation_order_resolved,
        )
        warnings.extend(result.warnings)
        final_map = _allocated_to_map(result.allocated, t2)

    report(98, "Writing output")
    raster_io.write_categorical_raster(final_map, t2.profile, cfg.output_file, nodata=t2.nodata)
    logger.info("--- LULC Prediction complete ---")
    report(100, "Complete")

    return PipelineResult(
        output_file=cfg.output_file,
        transition_matrix=tm,
        yearly_transition_matrix=new_tm,
        model_summary=model_summary,
        step_files=step_files,
        suitability_files=suitability_files,
        warnings=warnings,
    )


def build_run_config(
    modelType: Union[str, Sequence[str]],
    T1File: str,
    T2File: str,
    withClassName: Sequence[str],
    T1drivers: Dict[str, str],
    T2drivers: Dict[str, str],
    na_value: Optional[float],
    demand: Optional[Sequence[float]],
    restrictSpatialMigration: Optional[Sequence[float]],
    neighbour: Optional[Sequence[int]],
    outputfile: str,
    conversionOrder: Union[str, Sequence[Sequence[int]]] = "TP",
    classAllocationOrder: Optional[Sequence[int]] = None,
    maskFile: Optional[str] = None,
    aoiFile: Optional[str] = None,
    modelformula: Optional[Sequence[Optional[str]]] = None,
    suitabilityFileDirectory: Optional[str] = None,
    method: str = "NotIncludeCurrentClass",
) -> RunConfig:
    """Flat R-derived arguments -> :class:`RunConfig`, applying every
    unset-means-default rule in one place (model type broadcast, inertia
    defaulting to 0.0 per class, demand/model_formula/neighbourhood/
    class_allocation_order defaulting to None/unset). Shared by
    :func:`generate_predicted_map` (the GUI's calling convention) and
    :class:`LULC.scenario.ScenarioFile` (YAML scenario files, GUI Open/Save
    and ``OpenLDM.py --config``) so both go through the exact same defaults
    rather than each re-implementing this fill-in logic separately."""
    class_ids = get_class_codes(T1File, na_value=na_value)
    if len(class_ids) != len(withClassName):
        raise DatasetValidationError(
            f"T1 raster has {len(class_ids)} classes {class_ids} but "
            f"{len(withClassName)} class names were supplied"
        )

    n = len(class_ids)
    model_types = [modelType] * n if isinstance(modelType, str) else list(modelType)
    formulas = list(modelformula) if modelformula is not None else [None] * n
    inertia = list(restrictSpatialMigration) if restrictSpatialMigration is not None else [0.0] * n
    demands = list(demand) if demand is not None else [None] * n

    classes = [
        ClassConfig(
            name=withClassName[i],
            class_id=class_ids[i],
            model_type=model_types[i],
            model_formula=formulas[i],
            inertia=inertia[i],
            demand=demands[i],
        )
        for i in range(n)
    ]

    neighbourhood = None
    if neighbour is not None:
        neighbourhood = NeighbourhoodConfig(
            window_size=int(neighbour[0]),
            steps=int(neighbour[1]),
            write_step_output=bool(int(neighbour[2])) if len(neighbour) > 2 else False,
        )

    return RunConfig(
        t1_file=T1File,
        t2_file=T2File,
        classes=classes,
        t1_drivers=dict(T1drivers),
        t2_drivers=dict(T2drivers),
        output_file=outputfile,
        na_value=na_value,
        conversion_order=conversionOrder,
        class_allocation_order=classAllocationOrder,
        neighbourhood=neighbourhood,
        method=method,
        mask_file=maskFile,
        aoi_file=aoiFile,
        suitability_file_directory=suitabilityFileDirectory,
    )


def generate_predicted_map(
    modelType: Union[str, Sequence[str]],
    T1File: str,
    T2File: str,
    withClassName: Sequence[str],
    T1drivers: Dict[str, str],
    T2drivers: Dict[str, str],
    na_value: Optional[float],
    demand: Optional[Sequence[float]],
    restrictSpatialMigration: Optional[Sequence[float]],
    neighbour: Optional[Sequence[int]],
    outputfile: str,
    conversionOrder: Union[str, Sequence[Sequence[int]]] = "TP",
    classAllocationOrder: Optional[Sequence[int]] = None,
    maskFile: Optional[str] = None,
    aoiFile: Optional[str] = None,
    modelformula: Optional[Sequence[Optional[str]]] = None,
    suitabilityFileDirectory: Optional[str] = None,
    AllowedClassMigration: Optional[Sequence[Sequence[int]]] = None,
    method: str = "NotIncludeCurrentClass",
    on_progress: Optional[Callable[[int, str], None]] = None,
) -> PipelineResult:
    """Backward-compatible flat-argument wrapper around :func:`run_pipeline`.

    Mirrors R's ``genratePredictedMap`` signature (and the pre-Slice-1 Python
    facade). Class codes are taken from the T1 raster in sorted order and
    paired positionally with ``withClassName`` — the R backend's convention.
    """
    if AllowedClassMigration is not None:
        # TODO(slice-2): R applies this matrix positionally onto the derived
        # conversion order, which is almost certainly not the intended
        # semantics; the GUI always passes NA. Deferred.
        raise NotImplementedError("AllowedClassMigration is not supported in this slice")

    cfg = build_run_config(
        modelType=modelType,
        T1File=T1File,
        T2File=T2File,
        withClassName=withClassName,
        T1drivers=T1drivers,
        T2drivers=T2drivers,
        na_value=na_value,
        demand=demand,
        restrictSpatialMigration=restrictSpatialMigration,
        neighbour=neighbour,
        outputfile=outputfile,
        conversionOrder=conversionOrder,
        classAllocationOrder=classAllocationOrder,
        maskFile=maskFile,
        aoiFile=aoiFile,
        modelformula=modelformula,
        suitabilityFileDirectory=suitabilityFileDirectory,
        method=method,
    )
    return run_pipeline(cfg, on_progress=on_progress)


def get_kappa_summary(
    actualFile: str,
    predictedFile: str,
    na_value: Optional[float] = None,
    classNames: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Accuracy assessment of a predicted map against a reference map.

    Port of R's ``getKappaSummary`` (lines 2375-2387): confusion matrix via
    ``createTM``, then the classic kappa statistics. (R's version crashes on
    an undefined variable; this implementation is the intended
    behavior.) Returns the confusion matrix, a :class:`accuracy.KappaStatistics`,
    and the GUI-parseable ``~~`` summary string.
    """
    logger.info("--- Calculating Accuracy ---")
    actual = raster_io.read_categorical_raster(actualFile, na_value=na_value)
    predicted = raster_io.read_categorical_raster(predictedFile, na_value=na_value)

    actual_codes = np.where(actual.valid_mask(), actual.single_band, np.nan)
    predicted_codes = np.where(predicted.valid_mask(), predicted.single_band, np.nan)

    ids = sorted(
        set(np.unique(actual_codes[np.isfinite(actual_codes)]).astype(int))
        | set(np.unique(predicted_codes[np.isfinite(predicted_codes)]).astype(int))
    )
    cm = transition.build_transition_matrix(actual_codes, predicted_codes, ids)
    stats = accuracy.kappa_statistics(cm)

    logger.info("\n" + stats.summary_text())
    return {
        "confusion_matrix": cm,
        "class_ids": ids,
        "class_names": list(classNames) if classNames is not None else None,
        "statistics": stats,
        "accuracy": stats.sum_naive,
        "kappa": stats.sum_kappa,
        "py_kappa_summary": stats.py_kappa_summary(),
    }


def kappa_agreement_index(
    simulationFile: str,
    actualFile: str,
    baseFile: Optional[str] = None,
    na_value: Optional[float] = None,
) -> pontius.PontiusAgreementIndex:
    """Port of R's ``kappa.agreementindex``: the Pontius/Van Vliet
    three-map disagreement decomposition, comparing a simulated map against
    the actual map it was meant to predict, conditioned on a base-year map.
    See ``LULC.pontius`` for the full formula-by-formula port; this is a
    thin facade, matching :func:`get_kappa_summary`'s pattern for the
    simpler two-map kappa chain. ``baseFile`` is optional — see
    ``pontius.kappa_agreement_index``'s docstring for what's skipped
    without one.
    """
    return pontius.kappa_agreement_index(simulationFile, actualFile, baseFile, na_value=na_value)


def generate_map(
    plotFile: str,
    outputFile: str,
    classNumber: Sequence[int],
    className: Sequence[str],
    classColour: Sequence[str],
    plotTitle: str = "",
    plotLegendTitle: str = "",
    na_value: Optional[float] = None,
) -> str:
    """Port of R's ``genrateMap``: render ``plotFile`` as a
    colored, legended map (discrete per-class legend, scale bar, north
    arrow, title) and write it to ``outputFile``. See ``LULC.cartography``
    for the full implementation; this is a thin facade, matching
    :func:`kappa_agreement_index`'s pattern. Returns ``outputFile``.
    """
    classes = [
        cartography.ClassStyle(class_id=cid, name=name, color=color)
        for cid, name, color in zip(classNumber, className, classColour)
    ]
    fig = cartography.render_classified_map(
        plotFile, classes, title=plotTitle, legend_title=plotLegendTitle, na_value=na_value,
    )
    cartography.save_map(fig, outputFile)
    return outputFile


def get_model_fit_summary(
    T1File: str,
    T2File: str,
    T1drivers: Dict[str, str],
    modelType: Union[str, Sequence[str]],
    withNAvalue: Optional[float] = None,
    method: str = "NotIncludeCurrentClass",
    maskFile: Optional[str] = None,
    aoiFile: Optional[str] = None,
) -> Dict[str, str]:
    """Fit models and return their text summaries (R: getModelFitSummary).

    Used by the GUI's "View Model Statistics" button; T2File is accepted for
    signature parity (R loads it but the summary only reflects the T1 fit).
    """
    t1 = masking.apply_mask_and_aoi(
        raster_io.read_categorical_raster(T1File, na_value=withNAvalue), maskFile, aoiFile
    )
    drivers = masking.apply_mask_and_aoi(
        raster_io.read_driver_stack(T1drivers), maskFile, aoiFile
    )
    t1_codes = np.where(t1.valid_mask(), t1.single_band, np.nan)
    class_ids = [int(c) for c in np.unique(t1_codes[np.isfinite(t1_codes)])]
    class_names = [f"class_{c}" for c in class_ids]
    n = len(class_ids)
    model_types = [modelType] * n if isinstance(modelType, str) else list(modelType)

    models = modeling.fit_models_separately(
        t1_codes=t1_codes,
        driver_stack_t1=np.where(drivers.valid_mask(), drivers.array, np.nan),
        driver_names=drivers.band_names,
        class_ids=class_ids,
        class_names=class_names,
        model_types=model_types,
        method=method,
    )
    return modeling.get_model_summary(models)
