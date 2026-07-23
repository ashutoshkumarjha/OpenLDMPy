"""Unit tests for gui.controller.PipelineController — no real Qt widgets
needed (build_run_config takes plain Python values)."""

import rasterio as rio

from gui.controller import PipelineController
from LULC.config import NeighbourhoodConfig


def test_build_run_config_basic_fields(t1_file, t2_file, drivers_85, drivers_95, class_names, tmp_path):
    controller = PipelineController()
    class_ids = list(range(1, 10))
    cfg = controller.build_run_config(
        t1_file=t1_file,
        t2_file=t2_file,
        class_names=class_names,
        class_ids=class_ids,
        t1_drivers=drivers_85,
        t2_drivers=drivers_95,
        output_file=str(tmp_path / "out.tif"),
        model_types="logistic",
        demand=[1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992],
        inertia=[1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76],
        class_allocation_order=[1, 3, 4, 5, 9, 6, 7, 8, 2],
        window_size=3,
        steps=2,
        write_step_output=True,
    )
    assert cfg.class_names == class_names
    assert [c.class_id for c in cfg.classes] == class_ids
    assert cfg.model_types == ["logistic"] * 9
    assert cfg.demand == [1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992]
    assert cfg.restrict_spatial_migration == [1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76]
    assert cfg.class_allocation_order == [1, 3, 4, 5, 9, 6, 7, 8, 2]
    assert isinstance(cfg.neighbourhood, NeighbourhoodConfig)
    assert cfg.neighbourhood.window_size == 3
    assert cfg.neighbourhood.steps == 2
    assert cfg.neighbourhood.write_step_output is True


def test_build_run_config_defaults_without_optional_fields(t1_file, t2_file, drivers_85, drivers_95, class_names, tmp_path):
    controller = PipelineController()
    cfg = controller.build_run_config(
        t1_file=t1_file,
        t2_file=t2_file,
        class_names=class_names,
        class_ids=list(range(1, 10)),
        t1_drivers=drivers_85,
        t2_drivers=drivers_95,
        output_file=str(tmp_path / "out.tif"),
        model_types="logistic",
    )
    assert cfg.demand is None
    assert cfg.restrict_spatial_migration == [0.0] * 9
    assert cfg.neighbourhood is None
    assert cfg.conversion_order == "TP"


def test_build_run_config_rejects_mismatched_lengths(t1_file, t2_file, drivers_85, drivers_95, tmp_path):
    controller = PipelineController()
    try:
        controller.build_run_config(
            t1_file=t1_file,
            t2_file=t2_file,
            class_names=["A", "B"],
            class_ids=[1, 2, 3],
            t1_drivers=drivers_85,
            t2_drivers=drivers_95,
            output_file=str(tmp_path / "out.tif"),
            model_types="logistic",
        )
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_get_class_codes(t1_file):
    controller = PipelineController()
    assert controller.get_class_codes(t1_file) == list(range(1, 10))


def test_run_prediction_from_gui_state_matches_flat_facade(
    t1_file, t2_file, drivers_85, drivers_95, class_names, tmp_path
):
    """The GUI's Execute path (run_prediction_from_gui_state) must behave
    identically to the flat generate_predicted_map facade it wraps —
    including honoring class_allocation_order (a real bug fix)."""
    controller = PipelineController()
    output_file = str(tmp_path / "predicted.tif")
    result = controller.run_prediction_from_gui_state(
        model_type=["logistic"] * 9,
        t1_file=t1_file,
        t2_file=t2_file,
        class_names=class_names,
        t1_drivers=drivers_85,
        t2_drivers=drivers_95,
        na_value=None,
        demand=[1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992],
        restrict_spatial_migration=[1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76],
        neighbour=None,
        output_file=output_file,
        conversion_order="TP",
        class_allocation_order=[1, 3, 4, 5, 9, 6, 7, 8, 2],
        model_formula=None,
        mask_file=None,
        aoi_file=None,
        suitability_file_directory=None,
    )
    with rio.open(output_file) as src:
        assert src.read(1).size > 0
    assert result.transition_matrix.shape == (9, 9)


def test_get_kappa_summary(t1_file, t3_file, class_names):
    controller = PipelineController()
    summary = controller.get_kappa_summary(t1_file, t3_file, class_names=class_names)
    assert "statistics" in summary
    assert -1.0 <= summary["kappa"] <= 1.0
