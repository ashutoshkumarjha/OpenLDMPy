"""YAML scenario files: a human-readable, re-loadable parameters file for a
single prediction run.

Replaces two things that used to exist separately and didn't round-trip
with each other:

* The GUI's "Save File" used to export a standalone, runnable ``.py``
  script with every value baked in as a literal (originally a runnable
  ``.R`` script in the R version) — write-only, nothing could read it back.
* ``OpenLDM.py`` used to be a fully hardcoded scenario with no way to point it
  at different data without editing the source.

:class:`ScenarioFile` is the single format both directions now go through:
the GUI's Save/Open menu actions write and read it (``gui/main_window.py``),
and ``OpenLDM.py --config`` loads it for headless runs. Any field left unset
("placeholder", matching a GUI checkbox left unchecked) is written as YAML
``null`` and resolved to a real default by :meth:`ScenarioFile.to_run_config`
via :func:`LULC.LULCAlgorithms.build_run_config` — the same default-filling
logic ``generate_predicted_map`` (the GUI's own execution path) already
uses, so a scenario loaded from a partially-filled file behaves identically
to running the GUI with those same fields left at their defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Union

import yaml

from . import LULCAlgorithms
from .config import RunConfig
from .errors import PipelineError

SCENARIO_VERSION = 1


class ScenarioFileError(PipelineError):
    """Raised for a malformed or unsupported scenario YAML file."""


@dataclass
class ScenarioClass:
    """One row of the ``classes:`` list. ``class_id`` is informational on
    load (it's re-derived from the raster's own class codes by
    :meth:`ScenarioFile.to_run_config`, positionally paired with the class
    list in the same order R/the GUI has always used) but is written out on
    save so the file is self-describing.

    ``legend_text``/``colour`` are View Maps' ``twColorTable_ViewMaps``
    columns 1/2 (legend labels come from ``legend_text``,
    *not* ``name`` — that bug is exactly why the two are tracked
    separately here). ``null``/absent means "use the GUI's own default":
    a ``Class-<N>`` placeholder label and an auto-generated greyscale ramp,
    same as a freshly-built, never-edited color table."""

    name: Optional[str] = None
    class_id: Optional[int] = None
    model_type: str = "logistic"
    demand: Optional[float] = None
    inertia: Optional[float] = None
    legend_text: Optional[str] = None
    colour: Optional[str] = None  # "#rrggbb"


@dataclass
class ScenarioSpatialContext:
    enabled: bool = False
    window_size: int = 3
    steps: int = 1
    write_step_output: bool = False


@dataclass
class ScenarioAccuracyAssessment:
    """The Accuracy Assessment tab. ``reference_file`` is the "Actual
    File"; ``predicted_file`` is normally the same raster ``data.output_file``
    names (the GUI keeps them in sync automatically whenever Output File
    changes) but can be pointed at a different, already-generated raster
    independently — ``null`` here means "use data.output_file", matching
    that auto-sync. ``base_file`` is optional: unset means the Pontius
    ksimulation/ktransition/ktranslocation metrics are left blank,
    exactly like leaving the GUI's own "Base File (Optional)" field empty.
    ``display_mode`` ("classwise" or "overall") only selects which
    Agreement Index table view is showing — cosmetic, doesn't change what
    gets computed."""

    reference_file: Optional[str] = None
    predicted_file: Optional[str] = None
    base_file: Optional[str] = None
    display_mode: str = "classwise"


@dataclass
class ScenarioMapComposition:
    """The View Maps tab. ``source_file`` is which raster Show/Export
    renders; ``null`` means "use data.output_file" (the run's own
    prediction — the common case of viewing what you just generated).
    ``export_file`` is where Export would write the PNG; the GUI doesn't
    track a "last export path" anywhere on its own (each Export click
    prompts fresh via a save dialog), so Save leaves it unset — it exists
    here for a hand-edited file to pre-fill that prompt, or for scripted/
    future direct-render use, not because the GUI itself round-trips it."""

    source_file: Optional[str] = None
    export_file: Optional[str] = None
    title: str = ""
    legend_heading: str = ""


@dataclass
class ScenarioFile:
    """Full parameterization of a single run, superset of :class:`RunConfig`
    with the extra fields the GUI needs to restore itself (not needed to
    actually run the pipeline): ``t1_year``/``t2_year`` (drive the Spatial
    Context "In Steps" combo's range), plus the Accuracy Assessment and
    View Maps tabs (:class:`ScenarioAccuracyAssessment`/
    :class:`ScenarioMapComposition` — neither is a `run_pipeline` input;
    both describe what happens to the run's *output*, after Execute)."""

    t1_file: Optional[str] = None
    t1_year: Optional[int] = None
    t2_file: Optional[str] = None
    t2_year: Optional[int] = None
    output_file: Optional[str] = None
    na_value: Optional[float] = None
    area_of_interest_file: Optional[str] = None
    mask_file: Optional[str] = None

    drivers_t1: Dict[str, str] = field(default_factory=dict)
    drivers_t2: Dict[str, str] = field(default_factory=dict)

    classes: List[ScenarioClass] = field(default_factory=list)

    class_allocation_order: Optional[Sequence[int]] = None
    conversion_order: Union[str, Sequence[Sequence[int]]] = "TP"

    spatial_context: ScenarioSpatialContext = field(default_factory=ScenarioSpatialContext)

    suitability_file_directory: Optional[str] = None

    accuracy_assessment: ScenarioAccuracyAssessment = field(default_factory=ScenarioAccuracyAssessment)
    map_composition: ScenarioMapComposition = field(default_factory=ScenarioMapComposition)

    # ------------------------------------------------------------------
    # YAML I/O
    # ------------------------------------------------------------------

    def to_yaml(self, path: str) -> None:
        doc = {
            "scenario_version": SCENARIO_VERSION,
            "data": {
                "t1_file": self.t1_file,
                "t1_year": self.t1_year,
                "t2_file": self.t2_file,
                "t2_year": self.t2_year,
                "output_file": self.output_file,
                "na_value": self.na_value,
                "area_of_interest_file": self.area_of_interest_file,
                "mask_file": self.mask_file,
            },
            "drivers": {
                "t1": dict(self.drivers_t1),
                "t2": dict(self.drivers_t2),
            },
            "classes": [
                {
                    "name": c.name,
                    "class_id": c.class_id,
                    "model_type": c.model_type,
                    "demand": c.demand,
                    "inertia": c.inertia,
                    "legend_text": c.legend_text,
                    "colour": c.colour,
                }
                for c in self.classes
            ]
            or None,
            "allocation": {
                "class_allocation_order": (
                    list(self.class_allocation_order) if self.class_allocation_order is not None else None
                ),
                "conversion_order": self.conversion_order,
            },
            "spatial_context": {
                "enabled": self.spatial_context.enabled,
                "window_size": self.spatial_context.window_size,
                "steps": self.spatial_context.steps,
                "write_step_output": self.spatial_context.write_step_output,
            },
            "suitability_file_directory": self.suitability_file_directory,
            "accuracy_assessment": {
                "reference_file": self.accuracy_assessment.reference_file,
                "predicted_file": self.accuracy_assessment.predicted_file,
                "base_file": self.accuracy_assessment.base_file,
                "display_mode": self.accuracy_assessment.display_mode,
            },
            "map_composition": {
                "source_file": self.map_composition.source_file,
                "export_file": self.map_composition.export_file,
                "title": self.map_composition.title,
                "legend_heading": self.map_composition.legend_heading,
            },
        }
        with open(path, "w") as fh:
            yaml.safe_dump(doc, fh, sort_keys=False, default_flow_style=False)

    @classmethod
    def from_yaml(cls, path: str) -> "ScenarioFile":
        """Raises :class:`ScenarioFileError` uniformly for every "this file
        isn't usable" reason — missing/unreadable file, invalid YAML
        syntax, a YAML document that isn't a mapping at the top level, or
        an unsupported ``scenario_version`` — so callers (GUI Open,
        ``OpenLDM.py --config``) only ever need to catch the one exception
        type to handle a malformed/corrupted config file."""
        try:
            with open(path) as fh:
                doc = yaml.safe_load(fh)
        except OSError as exc:
            raise ScenarioFileError(f"{path}: could not be read ({exc})") from exc
        except yaml.YAMLError as exc:
            raise ScenarioFileError(f"{path}: not valid YAML ({exc})") from exc

        if doc is None:
            doc = {}
        if not isinstance(doc, dict):
            raise ScenarioFileError(
                f"{path}: expected a YAML mapping at the top level, got {type(doc).__name__}"
            )

        version = doc.get("scenario_version")
        if version != SCENARIO_VERSION:
            raise ScenarioFileError(
                f"{path}: unsupported scenario_version {version!r} "
                f"(expected {SCENARIO_VERSION})"
            )

        try:
            data = doc.get("data") or {}
            drivers = doc.get("drivers") or {}
            allocation = doc.get("allocation") or {}
            spatial = doc.get("spatial_context") or {}
            accuracy = doc.get("accuracy_assessment") or {}
            map_comp = doc.get("map_composition") or {}
            classes = [
                ScenarioClass(
                    name=c.get("name"),
                    class_id=c.get("class_id"),
                    model_type=c.get("model_type") or "logistic",
                    demand=c.get("demand"),
                    inertia=c.get("inertia"),
                    legend_text=c.get("legend_text"),
                    colour=c.get("colour"),
                )
                for c in (doc.get("classes") or [])
            ]

            return cls(
                t1_file=data.get("t1_file"),
                t1_year=data.get("t1_year"),
                t2_file=data.get("t2_file"),
                t2_year=data.get("t2_year"),
                output_file=data.get("output_file"),
                na_value=data.get("na_value"),
                area_of_interest_file=data.get("area_of_interest_file"),
                mask_file=data.get("mask_file"),
                drivers_t1=dict(drivers.get("t1") or {}),
                drivers_t2=dict(drivers.get("t2") or {}),
                classes=classes,
                class_allocation_order=allocation.get("class_allocation_order"),
                conversion_order=allocation.get("conversion_order", "TP"),
                spatial_context=ScenarioSpatialContext(
                    enabled=bool(spatial.get("enabled", False)),
                    window_size=int(spatial.get("window_size", 3)),
                    steps=int(spatial.get("steps", 1)),
                    write_step_output=bool(spatial.get("write_step_output", False)),
                ),
                suitability_file_directory=doc.get("suitability_file_directory"),
                accuracy_assessment=ScenarioAccuracyAssessment(
                    reference_file=accuracy.get("reference_file"),
                    predicted_file=accuracy.get("predicted_file"),
                    base_file=accuracy.get("base_file"),
                    display_mode=accuracy.get("display_mode") or "classwise",
                ),
                map_composition=ScenarioMapComposition(
                    source_file=map_comp.get("source_file"),
                    export_file=map_comp.get("export_file"),
                    title=map_comp.get("title") or "",
                    legend_heading=map_comp.get("legend_heading") or "",
                ),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            # A section present but the wrong shape (e.g. "data: not a
            # mapping", "classes: not a list of mappings") -- everything
            # else about the file was well-formed enough to get here, but
            # this is still "not usable", same as a syntax error.
            raise ScenarioFileError(f"{path}: malformed scenario structure ({exc})") from exc

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def to_run_config(self) -> RunConfig:
        """Resolve every placeholder to its real default and build a
        :class:`RunConfig`, via :func:`LULCAlgorithms.build_run_config` — the
        same defaults ``generate_predicted_map``/the GUI's Execute path
        already apply, so a field left unset here behaves exactly as it
        would from an unchecked GUI checkbox."""
        if not self.t1_file or not self.t2_file or not self.output_file:
            raise ScenarioFileError(
                "scenario is missing one of the required fields: "
                "data.t1_file, data.t2_file, data.output_file"
            )

        if self.classes:
            class_names = [c.name or f"Class{c.class_id}" for c in self.classes]
            model_types = [c.model_type or "logistic" for c in self.classes]
            demand = (
                None
                if all(c.demand is None for c in self.classes)
                else [c.demand if c.demand is not None else 0 for c in self.classes]
            )
            inertia = (
                None
                if all(c.inertia is None for c in self.classes)
                else [c.inertia if c.inertia is not None else 0.0 for c in self.classes]
            )
        else:
            # No classes[] at all: derive names straight from the raster's
            # own class codes (same "not yet customized" convention the GUI
            # itself uses — see main_window.py's on_pbNext_DriverSelectionT0
            # / self.__className = [str(c) for c in class_ids]).
            class_ids = LULCAlgorithms.get_class_codes(self.t1_file, na_value=self.na_value)
            class_names = [str(c) for c in class_ids]
            model_types = "logistic"
            demand = None
            inertia = None

        neighbour = None
        if self.spatial_context.enabled:
            neighbour = [
                self.spatial_context.window_size,
                self.spatial_context.steps,
                1 if self.spatial_context.write_step_output else 0,
            ]

        return LULCAlgorithms.build_run_config(
            modelType=model_types,
            T1File=self.t1_file,
            T2File=self.t2_file,
            withClassName=class_names,
            T1drivers=self.drivers_t1,
            T2drivers=self.drivers_t2,
            na_value=self.na_value,
            demand=demand,
            restrictSpatialMigration=inertia,
            neighbour=neighbour,
            outputfile=self.output_file,
            conversionOrder=self.conversion_order,
            classAllocationOrder=(
                list(self.class_allocation_order) if self.class_allocation_order is not None else None
            ),
            maskFile=self.mask_file,
            aoiFile=self.area_of_interest_file,
            suitabilityFileDirectory=self.suitability_file_directory,
        )
