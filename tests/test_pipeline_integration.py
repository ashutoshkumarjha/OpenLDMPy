"""End-to-end pipeline test, standing in for src/runSteps1a.R.

Runs the full pure-Python prediction + accuracy pipeline against
data/example/ and checks the result is a sane, valid categorical raster —
not a bit-exact R comparison (the model-fitting stage is not deterministic
across R/sklearn implementations for randomForest/svm; the logistic branch
is checked separately and more precisely in test_modeling.py).
"""

import numpy as np
import rasterio as rio

from LULC import LULCAlgorithms as Algorithm
from LULC.config import ClassConfig, NeighbourhoodConfig, RunConfig


def test_generate_predicted_map_end_to_end(tmp_path, t1_file, t2_file, t3_file, drivers_85, drivers_95, class_names):
    output_file = str(tmp_path / "predicted.tif")
    driver_terms = "TD1.DistanceToDrainage+TD1.DistanceToBuiltup+TD1.DistanceToRoad+TD1.Elevation"
    model_formula = [f"T1.{name} ~ {driver_terms}" for name in class_names]

    progress_calls = []
    result = Algorithm.generate_predicted_map(
        modelType=["logistic"] * 9,
        T1File=t1_file,
        T2File=t2_file,
        withClassName=class_names,
        T1drivers=drivers_85,
        T2drivers=drivers_95,
        na_value=None,
        demand=[1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992],
        restrictSpatialMigration=[1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76],
        neighbour=None,  # single-shot branch, faster for the test suite
        outputfile=output_file,
        conversionOrder="TP",
        classAllocationOrder=[1, 3, 4, 5, 9, 6, 7, 8, 2],
        maskFile=None,
        aoiFile=None,
        modelformula=model_formula,
        suitabilityFileDirectory=None,
        on_progress=lambda pct, label: progress_calls.append((pct, label)),
    )

    assert result.output_file == output_file
    with rio.open(output_file) as src:
        data = src.read(1)
        nodata = src.nodata
        valid = data[data != nodata]
        assert valid.size > 0
        assert set(np.unique(valid).astype(int)).issubset(set(range(1, 10)))

    assert result.transition_matrix.shape == (9, 9)

    # Real progress callback (gui/progress_bridge.py's counterpart): called
    # with monotonically increasing, bounded percentages, ending at 100.
    percents = [p for p, _ in progress_calls]
    assert percents, "on_progress was never called"
    assert all(0 <= p <= 100 for p in percents)
    assert percents == sorted(percents)
    assert percents[-1] == 100


def test_multi_step_neighbourhood_branch_runs(tmp_path, t1_file, t2_file, drivers_85, drivers_95, class_names):
    """Smaller/faster check that Branch B (windowed multi-step simulation)
    executes and writes step files, without asserting on exact class counts."""
    output_file = str(tmp_path / "predicted_steps.tif")
    cfg = RunConfig(
        t1_file=t1_file,
        t2_file=t2_file,
        classes=[
            ClassConfig(name=n, class_id=i + 1, model_type="logistic", inertia=r)
            for i, (n, r) in enumerate(
                zip(class_names, [1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76])
            )
        ],
        t1_drivers=drivers_85,
        t2_drivers=drivers_95,
        output_file=output_file,
        na_value=None,  # sample rasters carry their own float nodata, not the 128 default
        conversion_order="TP",
        class_allocation_order=[1, 3, 4, 5, 9, 6, 7, 8, 2],
        neighbourhood=NeighbourhoodConfig(window_size=3, steps=2, write_step_output=True),
    )
    result = Algorithm.run_pipeline(cfg)
    assert len(result.step_files) == 2
    for path in result.step_files:
        with rio.open(path) as src:
            assert src.read(1).size > 0


def test_mixed_model_types_end_to_end(tmp_path, t1_file, t2_file, drivers_85, drivers_95, class_names):
    """Exercise every supported model_type (logistic, randomForest, svm,
    nnet) together in one run, cycling across the 9 classes, mirroring the
    mixed scenario commented out in src/runSteps1a.R (model.type=c(
    'randomForest',...,'logistic',...,'logistic') -> output file suffix
    "rrrrrrlrl") but going further to also cover svm/nnet in the same run."""
    output_file = str(tmp_path / "predicted_mixed.tif")
    model_types = ["randomForest", "logistic", "svm", "nnet", "randomForest", "logistic", "svm", "nnet", "randomForest"]
    assert len(model_types) == len(class_names)

    result = Algorithm.generate_predicted_map(
        modelType=model_types,
        T1File=t1_file,
        T2File=t2_file,
        withClassName=class_names,
        T1drivers=drivers_85,
        T2drivers=drivers_95,
        na_value=None,
        demand=[1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992],
        restrictSpatialMigration=[1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76],
        neighbour=None,
        outputfile=output_file,
        conversionOrder="TP",
        classAllocationOrder=[1, 3, 4, 5, 9, 6, 7, 8, 2],
        maskFile=None,
        aoiFile=None,
        modelformula=None,  # no formula override -> uses all drivers, per-class model_type still varies
        suitabilityFileDirectory=None,
    )

    with rio.open(output_file) as src:
        data = src.read(1)
        nodata = src.nodata
        valid = data[data != nodata]
        assert valid.size > 0
        assert set(np.unique(valid).astype(int)).issubset(set(range(1, 10)))

    # Every model type actually got exercised (not silently defaulted to
    # logistic) — get_model_summary embeds "Model: <type>" in each entry.
    assert result.model_summary.keys() == set(class_names)
    for name, model_type in zip(class_names, model_types):
        assert f"Model: {model_type}" in result.model_summary[name]


def test_kappa_summary_end_to_end(t1_file, t3_file, class_names):
    summary = Algorithm.get_kappa_summary(
        actualFile=t3_file, predictedFile=t1_file, na_value=None, classNames=class_names
    )
    assert -1.0 <= summary["kappa"] <= 1.0
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert "~~" in summary["py_kappa_summary"]
