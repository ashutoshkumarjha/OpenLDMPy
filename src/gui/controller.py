"""Controller layer: translates GUI state into calls on the ``LULC``
processing package.

This is the "Controller Layer" node (Job Manager / Parameter Validation)
from the project's target architecture diagram. It is deliberately Qt-light
— ``build_run_config`` takes plain Python values (whatever the main window's
``prepare*`` methods extract from its widgets), not widgets themselves, so
it can be unit-tested without instantiating real Qt objects and so it is
the natural seam a future QGIS Processing algorithm calls into instead of a
``QMainWindow`` (the second migration subtask).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Union

from LULC import LULCAlgorithms
from LULC.accuracy import DatasetCheckReport
from LULC.config import ClassConfig, NeighbourhoodConfig, PipelineResult, RunConfig


class PipelineController:
    """Thin, Qt-free wrapper around :mod:`LULC.LULCAlgorithms`."""

    def get_class_codes(self, t1_file: str, na_value: Optional[float] = None) -> List[int]:
        return LULCAlgorithms.get_class_codes(t1_file, na_value=na_value)

    def rasterize_shapefile(
        self,
        shp_file: str,
        output_file: str,
        grid_size: float = 1000.0,
        option: str = "fraction",
        class_field: str = "IGBP_CODE",
        poly_id_field: Optional[str] = None,
    ) -> str:
        return LULCAlgorithms.rasterize_shapefile(
            shp_file, output_file, class_field=class_field,
            grid_size=grid_size, option=option, poly_id_field=poly_id_field,
        )

    def is_dataset_correct(
        self,
        t1_drivers: Dict[str, str],
        t2_drivers: Dict[str, str],
        t1_file: str,
        t2_file: str,
        t3_file: Optional[str] = None,
    ) -> bool:
        return LULCAlgorithms.is_dataset_correct(t1_drivers, t2_drivers, t1_file, t2_file, t3_file)

    def check_dataset(
        self,
        t1_drivers: Dict[str, str],
        t2_drivers: Dict[str, str],
        t1_file: str,
        t2_file: str,
        t3_file: Optional[str] = None,
        extra_layers: Optional[Dict[str, str]] = None,
    ) -> DatasetCheckReport:
        return LULCAlgorithms.check_dataset(
            t1_drivers, t2_drivers, t1_file, t2_file, t3_file, extra_layers=extra_layers,
        )

    def kappa_agreement_index(
        self,
        simulation_file: str,
        actual_file: str,
        base_file: Optional[str] = None,
        na_value: Optional[float] = None,
    ):
        """Pontius/Van Vliet three-map disagreement decomposition.
        Wired to the Accuracy Assessment tab's "Agreement Index" radio
        buttons; ``base_file`` is optional there too — see
        ``pontius.kappa_agreement_index``'s docstring."""
        return LULCAlgorithms.kappa_agreement_index(
            simulation_file, actual_file, base_file, na_value=na_value,
        )

    def generate_map(
        self,
        plot_file: str,
        output_file: str,
        class_numbers: Sequence[int],
        class_names: Sequence[str],
        class_colours: Sequence[str],
        title: str = "",
        legend_title: str = "",
        na_value: Optional[float] = None,
    ) -> str:
        """Cartographic rendering. Wired to the View Maps tab's
        Show/Export buttons."""
        return LULCAlgorithms.generate_map(
            plot_file, output_file, class_numbers, class_names, class_colours,
            plotTitle=title, plotLegendTitle=legend_title, na_value=na_value,
        )

    def get_model_fit_summary(
        self,
        t1_file: str,
        t2_file: str,
        t1_drivers: Dict[str, str],
        model_type: Union[str, Sequence[str]],
        na_value: Optional[float] = None,
        method: str = "NotIncludeCurrentClass",
        mask_file: Optional[str] = None,
        aoi_file: Optional[str] = None,
    ) -> Dict[str, str]:
        return LULCAlgorithms.get_model_fit_summary(
            T1File=t1_file,
            T2File=t2_file,
            T1drivers=t1_drivers,
            modelType=model_type,
            withNAvalue=na_value,
            method=method,
            maskFile=mask_file,
            aoiFile=aoi_file,
        )

    def build_run_config(
        self,
        t1_file: str,
        t2_file: str,
        class_names: Sequence[str],
        class_ids: Sequence[int],
        t1_drivers: Dict[str, str],
        t2_drivers: Dict[str, str],
        output_file: str,
        model_types: Union[str, Sequence[str]],
        na_value: Optional[float] = None,
        demand: Optional[Sequence[Optional[float]]] = None,
        inertia: Optional[Sequence[float]] = None,
        model_formulas: Optional[Sequence[Optional[str]]] = None,
        conversion_order: Union[str, Sequence[Sequence[int]]] = "TP",
        class_allocation_order: Optional[Sequence[int]] = None,
        window_size: Optional[int] = None,
        steps: int = 1,
        write_step_output: bool = False,
        mask_file: Optional[str] = None,
        aoi_file: Optional[str] = None,
        suitability_file_directory: Optional[str] = None,
    ) -> RunConfig:
        """Assemble a :class:`RunConfig` from plain, GUI-extracted values.

        Mirrors the original ``prepareDataForExecution``/``prepareSpatialData``/
        ``prepareDemand``/``prepareInertia``/``prepareConversionOrder``/
        ``prepareClassAllocationOrder`` methods, but produces one typed
        dataclass instead of a pile of R-vector-typed instance attributes.
        """
        n = len(class_names)
        if len(class_ids) != n:
            raise ValueError(f"{n} class names but {len(class_ids)} class ids")

        types = [model_types] * n if isinstance(model_types, str) else list(model_types)
        formulas = list(model_formulas) if model_formulas is not None else [None] * n
        inertias = list(inertia) if inertia is not None else [0.0] * n
        demands = list(demand) if demand is not None else [None] * n

        classes = [
            ClassConfig(
                name=class_names[i],
                class_id=class_ids[i],
                model_type=types[i],
                model_formula=formulas[i],
                inertia=inertias[i],
                demand=demands[i],
            )
            for i in range(n)
        ]

        neighbourhood = None
        if window_size is not None:
            neighbourhood = NeighbourhoodConfig(
                window_size=window_size, steps=steps, write_step_output=write_step_output
            )

        return RunConfig(
            t1_file=t1_file,
            t2_file=t2_file,
            classes=classes,
            t1_drivers=dict(t1_drivers),
            t2_drivers=dict(t2_drivers),
            output_file=output_file,
            na_value=na_value,
            conversion_order=conversion_order,
            class_allocation_order=class_allocation_order,
            neighbourhood=neighbourhood,
            mask_file=mask_file,
            aoi_file=aoi_file,
            suitability_file_directory=suitability_file_directory,
        )

    def run_prediction(self, config: RunConfig) -> PipelineResult:
        return LULCAlgorithms.run_pipeline(config)

    def run_prediction_from_gui_state(
        self,
        model_type: Union[str, Sequence[str]],
        t1_file: str,
        t2_file: str,
        class_names: Sequence[str],
        t1_drivers: Dict[str, str],
        t2_drivers: Dict[str, str],
        na_value: Optional[float],
        demand: Optional[Sequence[float]],
        restrict_spatial_migration: Optional[Sequence[float]],
        neighbour: Optional[Sequence[int]],
        output_file: str,
        conversion_order: Union[str, Sequence[Sequence[int]]],
        class_allocation_order: Optional[Sequence[int]],
        model_formula: Optional[Sequence[Optional[str]]],
        mask_file: Optional[str],
        aoi_file: Optional[str],
        suitability_file_directory: Optional[str],
        on_progress: Optional[Callable[[int, str], None]] = None,
    ) -> PipelineResult:
        """Mirrors R's ``genratePredictedMap`` / the original GUI's
        ``runRCommand`` flat-argument shape directly (class ids are derived
        from the T1 raster's sorted class codes, positionally paired with
        ``class_names`` — the GUI never tracks explicit class ids, only
        display labels, so :meth:`build_run_config` doesn't fit here).

        ``on_progress(percent, label)``, if given, is forwarded straight to
        LULCAlgorithms.generate_predicted_map/run_pipeline's own stage
        callback (see gui/progress_bridge.py).
        """
        return LULCAlgorithms.generate_predicted_map(
            modelType=model_type,
            T1File=t1_file,
            T2File=t2_file,
            withClassName=class_names,
            T1drivers=t1_drivers,
            T2drivers=t2_drivers,
            na_value=na_value,
            demand=demand,
            restrictSpatialMigration=restrict_spatial_migration,
            neighbour=neighbour,
            outputfile=output_file,
            conversionOrder=conversion_order,
            classAllocationOrder=class_allocation_order,
            maskFile=mask_file,
            aoiFile=aoi_file,
            modelformula=model_formula,
            suitabilityFileDirectory=suitability_file_directory,
            on_progress=on_progress,
        )

    def get_kappa_summary(
        self,
        actual_file: str,
        predicted_file: str,
        na_value: Optional[float] = None,
        class_names: Optional[Sequence[str]] = None,
    ) -> Dict[str, object]:
        return LULCAlgorithms.get_kappa_summary(
            actualFile=actual_file, predictedFile=predicted_file, na_value=na_value, classNames=class_names
        )
