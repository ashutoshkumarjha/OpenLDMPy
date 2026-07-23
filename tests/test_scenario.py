import pytest

from LULC.scenario import (
    ScenarioAccuracyAssessment,
    ScenarioClass,
    ScenarioFile,
    ScenarioFileError,
    ScenarioMapComposition,
    ScenarioSpatialContext,
)


def _full_scenario(t1_file, t2_file, output_file, drivers_85, drivers_95, class_names):
    return ScenarioFile(
        t1_file=t1_file,
        t1_year=1985,
        t2_file=t2_file,
        t2_year=1995,
        output_file=output_file,
        na_value=None,
        drivers_t1=dict(drivers_85),
        drivers_t2=dict(drivers_95),
        classes=[
            ScenarioClass(
                name=name, class_id=i + 1, model_type="logistic", demand=d, inertia=r,
                legend_text=f"{name} (legend)", colour=f"#{i:02x}{i:02x}{i:02x}",
            )
            for i, (name, d, r) in enumerate(zip(
                class_names,
                [1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992],
                [1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76],
            ))
        ],
        class_allocation_order=[1, 3, 4, 5, 9, 6, 7, 8, 2],
        conversion_order="TP",
        spatial_context=ScenarioSpatialContext(
            enabled=True, window_size=3, steps=2, write_step_output=True
        ),
        accuracy_assessment=ScenarioAccuracyAssessment(
            reference_file="/tmp/2005.tif",
            predicted_file=None,
            base_file="/tmp/1995.tif",
            display_mode="overall",
        ),
        map_composition=ScenarioMapComposition(
            source_file="/tmp/out.tif",
            export_file="/tmp/out.png",
            title="Predicted LULC 2005",
            legend_heading="Land Use Classes",
        ),
    )


def test_yaml_round_trip_preserves_all_fields(tmp_path, t1_file, t2_file, drivers_85, drivers_95, class_names):
    scenario = _full_scenario(t1_file, t2_file, "/tmp/out.tif", drivers_85, drivers_95, class_names)
    path = str(tmp_path / "scenario.yaml")
    scenario.to_yaml(path)

    loaded = ScenarioFile.from_yaml(path)

    assert loaded.t1_file == t1_file
    assert loaded.t1_year == 1985
    assert loaded.t2_file == t2_file
    assert loaded.t2_year == 1995
    assert loaded.output_file == "/tmp/out.tif"
    assert loaded.na_value is None
    assert loaded.drivers_t1 == drivers_85
    assert loaded.drivers_t2 == drivers_95
    assert [c.name for c in loaded.classes] == class_names
    assert [c.demand for c in loaded.classes] == [1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992]
    assert [c.inertia for c in loaded.classes] == [1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76]
    assert loaded.class_allocation_order == [1, 3, 4, 5, 9, 6, 7, 8, 2]
    assert loaded.conversion_order == "TP"
    assert loaded.spatial_context == ScenarioSpatialContext(
        enabled=True, window_size=3, steps=2, write_step_output=True
    )
    assert [c.legend_text for c in loaded.classes] == [f"{n} (legend)" for n in class_names]
    assert [c.colour for c in loaded.classes] == [f"#{i:02x}{i:02x}{i:02x}" for i in range(len(class_names))]
    assert loaded.accuracy_assessment == ScenarioAccuracyAssessment(
        reference_file="/tmp/2005.tif", predicted_file=None, base_file="/tmp/1995.tif", display_mode="overall",
    )
    assert loaded.map_composition == ScenarioMapComposition(
        source_file="/tmp/out.tif", export_file="/tmp/out.png",
        title="Predicted LULC 2005", legend_heading="Land Use Classes",
    )


def test_unset_fields_round_trip_as_none_placeholders(tmp_path, t1_file, t2_file):
    scenario = ScenarioFile(t1_file=t1_file, t2_file=t2_file, output_file="/tmp/out.tif")
    path = str(tmp_path / "scenario.yaml")
    scenario.to_yaml(path)

    loaded = ScenarioFile.from_yaml(path)

    assert loaded.classes == []
    assert loaded.class_allocation_order is None
    assert loaded.spatial_context.enabled is False
    assert loaded.mask_file is None
    assert loaded.area_of_interest_file is None
    assert loaded.accuracy_assessment == ScenarioAccuracyAssessment()
    assert loaded.map_composition == ScenarioMapComposition()


def test_from_yaml_rejects_unsupported_version(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("scenario_version: 999\ndata: {}\n")
    with pytest.raises(ScenarioFileError):
        ScenarioFile.from_yaml(str(path))


def test_from_yaml_rejects_missing_file():
    with pytest.raises(ScenarioFileError):
        ScenarioFile.from_yaml("/nonexistent/path/scenario.yaml")


def test_from_yaml_rejects_invalid_yaml_syntax(tmp_path):
    path = tmp_path / "bad.yaml"
    # Unbalanced brackets -- a real YAML syntax error, not just bad content.
    path.write_text("scenario_version: 1\ndata: [this is not closed\n")
    with pytest.raises(ScenarioFileError):
        ScenarioFile.from_yaml(str(path))


def test_from_yaml_rejects_non_mapping_top_level(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n")
    with pytest.raises(ScenarioFileError):
        ScenarioFile.from_yaml(str(path))


def test_from_yaml_rejects_malformed_section_shape(tmp_path):
    path = tmp_path / "bad.yaml"
    # "data" should be a mapping; a plain string breaks data.get(...).
    path.write_text("scenario_version: 1\ndata: not-a-mapping\n")
    with pytest.raises(ScenarioFileError):
        ScenarioFile.from_yaml(str(path))


def test_to_run_config_requires_core_files():
    scenario = ScenarioFile()
    with pytest.raises(ScenarioFileError):
        scenario.to_run_config()


def test_to_run_config_with_full_classes_applies_demand_inertia_allocation(
    t1_file, t2_file, drivers_85, drivers_95, class_names
):
    scenario = _full_scenario(t1_file, t2_file, "/tmp/out.tif", drivers_85, drivers_95, class_names)
    cfg = scenario.to_run_config()

    assert cfg.class_names == class_names
    assert cfg.demand == [1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992]
    assert cfg.restrict_spatial_migration == [1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76]
    assert cfg.class_allocation_order_resolved == [1, 3, 4, 5, 9, 6, 7, 8, 2]
    assert cfg.neighbourhood.window_size == 3
    assert cfg.neighbourhood.steps == 2
    assert cfg.neighbourhood.write_step_output is True


def test_to_run_config_without_classes_derives_names_from_raster(t1_file, t2_file, drivers_85, drivers_95):
    scenario = ScenarioFile(
        t1_file=t1_file, t2_file=t2_file, output_file="/tmp/out.tif",
        drivers_t1=drivers_85, drivers_t2=drivers_95,
    )
    cfg = scenario.to_run_config()

    # Placeholder default: numeric-string class names straight off the
    # raster's own class codes (same convention main_window.py uses).
    assert cfg.class_names == [str(i) for i in range(1, 10)]
    assert cfg.demand is None
    assert cfg.restrict_spatial_migration == [0.0] * 9
    assert cfg.class_allocation_order_resolved == list(range(1, 10))
    assert cfg.neighbourhood is None


def test_to_run_config_partial_demand_defaults_missing_entries_to_zero(
    t1_file, t2_file, drivers_85, drivers_95, class_names
):
    """A class list where only some rows have demand set is still an
    explicit 'Demand' scenario -- unset rows default to 0, matching
    main_window.py's prepareDemand() (blank cell while the Demand
    checkbox is checked -> 0, not None -- None is reserved for "Demand
    not user-defined at all")."""
    classes = [
        ScenarioClass(name=name, class_id=i + 1, demand=(100 if i == 0 else None))
        for i, name in enumerate(class_names)
    ]
    scenario = ScenarioFile(
        t1_file=t1_file, t2_file=t2_file, output_file="/tmp/out.tif",
        drivers_t1=drivers_85, drivers_t2=drivers_95, classes=classes,
    )
    cfg = scenario.to_run_config()

    assert cfg.classes[0].demand == 100
    assert cfg.classes[1].demand == 0
