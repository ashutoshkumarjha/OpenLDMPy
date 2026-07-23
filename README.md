# OpenLDMPy: Open-source Land-use Dynamics Modeling Platform (Pythonic)

**OpenLDM** is a high-performance, Python-based geospatial modeling platform designed to simulate and predict Land Use and Land Cover (LULC) changes. It integrates advanced Machine Learning algorithms with spatial neighborhood constraints and competitive demand allocation logic to project future landscape scenarios.It also has enhanced accracy and modeling sections.

> **Note:** This version marks a complete migration from the legacy R-based backend to a native Python architecture, utilizing `scikit-learn` for parallelized modeling and `rasterio` for efficient geospatial I/O.

---

## 🚀 Key Features

*   **Machine Learning Integration:** Train models using **Random Forest**, **SVM**, **Neural Networks (MLP)**, or **Logistic Regression** to predict land suitability based on driver variables (e.g., distance to roads, elevation, slope).
*   **High Performance:** Utilizes multi-core parallel processing (via `joblib`) for efficient model training and prediction on high-resolution rasters.
*   **Competitive Allocation Algorithm:** Implements an iterative, demand-driven allocation logic that handles complex competition between land classes based on suitability and transition matrices.
*   **Spatial Constraints:** Accounts for neighborhood effects, spatial inertia (persistence), and dynamic transition probabilities.
*   **Dual Interface, one entry point:** `OpenLDM.py --mode gui` (default) launches the user-friendly **PyQt5** interface for project setup, driver selection, and visual parameter configuration; `OpenLDM.py --mode nogui --config <scenario.yaml>` runs the same pipeline headlessly, optimized for HPC environments or batch processing.

---

> **Migration status:** the core numeric pipeline (data I/O, AOI/masking, transition matrices, model fitting, suitability prediction, neighborhood weighting, competitive allocation, accuracy assessment, Pontius agreement decomposition, cartographic map export, vector shapefile-to-raster conversion) and the Qt GUI are both fully migrated to pure Python with no `rpy2`/R runtime dependency.

## 🌿 Layout & Project Structure

Originally split by git branch, then briefly merged back as sibling `desktopapp/`/`plugin/` directories. The `desktopapp/` wrapper has since been flattened away: `src/`, `tests/`, `docs/`, and `data/` all live directly at the repository root now. Only `plugin/` remains a separate sibling directory.

* **Repository root** — the complete application described in this README:
  the pure-Python processing pipeline, the PyQt5 desktop GUI, and
  `OpenLDM.py`, the unified entry point.
* **`plugin/`** — the QGIS plugin: launches the same desktop GUI as a
  window inside QGIS, installing any missing Python dependencies into a
  self-contained plugin-local virtual environment on first run.
  `plugin/vendor/{LULC,gui,data,helpdoc}` are symlinks back to the
  directories below for local dev, dereferenced into real copies at
  deploy time so the shipped plugin is self-contained.

<details>
<summary>Click to expand full directory tree</summary>

```text
├── src/
│   ├── LULC/                     # Processing package (Qt-free)
│   │   ├── __init__.py             # Public API exports
│   │   ├── config.py               # RunConfig/ClassConfig dataclasses, shared logger
│   │   ├── errors.py                # PipelineError/DatasetValidationError/AllocationError
│   │   ├── raster_io.py            # RasterLayer + array-native raster I/O
│   │   ├── masking.py              # AOI/mask clipping (raster + shapefile)
│   │   ├── transition.py           # Transition-matrix / Markov chain logic
│   │   ├── spatial.py              # Neighborhood/focal operations
│   │   ├── modeling.py             # ML Engine (Scikit-Learn/Joblib)
│   │   ├── allocation.py           # Competitive Allocation Algorithm
│   │   ├── accuracy.py             # Kappa statistics, confusion matrix, dataset validation
│   │   ├── pontius.py               # Pontius agreement-index disagreement decomposition
│   │   ├── cartography.py           # Classified-map rendering (legend/scale bar/north arrow)
│   │   ├── rasterize.py             # Shapefile → raster conversion
│   │   ├── scenario.py              # ScenarioFile: YAML load/save for a run's parameters
│   │   └── LULCAlgorithms.py       # Main API Facade connecting Logic to GUI/CLI
│   │
│   ├── gui/                      # Qt GUI package — no rpy2/R
│   │   ├── main_window.py          # QMainWindow, ported from the legacy R-era GUI
│   │   ├── controller.py           # PipelineController — Qt-widget-state -> LULC calls
│   │   ├── workers.py              # BackgroundTaskWorker(QThread), keeps UI responsive
│   │   ├── log_bridge.py           # Forwards LULC's logger to the status bar live
│   │   ├── progress_bridge.py      # Forwards LULCAlgorithms' report(percent, label) to the progress bar
│   │   ├── OpenLDMgui.ui           # Qt Designer source for the GUI layout
│   │   ├── OpenLDMgui.qrc          # Qt resource collection (icons/images)
│   │   ├── OpenLDMgui.py           # pyuic5-generated from OpenLDMgui.ui — do not hand-edit
│   │   ├── OpenLDMgui_rc.py        # pyrcc5-generated from OpenLDMgui.qrc — do not hand-edit
│   │   └── images/                 # Icons/splash images baked into the .qrc
│   │
│   ├── helpdoc/                  # Help tab content (QWebEngineView), ported from the legacy GUI
│   └── OpenLDM.py                 # Unified entry point: --mode gui (default) or nogui
│
├── tests/                        # pytest suite, one file per LULC/gui module, incl. R-oracle comparisons
│   └── r_oracle/                   # Offline R fixture generator + checked-in reference CSVs
│
├── docs/                         # Architecture docs, function inventory, ADRs
│   └── adr/                        # Architecture Decision Records
│
├── data/example/                 # Sample Data Directory
│   ├── LULC/                       # Input Rasters (e.g., 1985.tif, 1995.tif)
│   ├── Drivers/                    # Spatial Drivers (dist. to road/stream, elevation)
│   ├── MaksFiles/                   # Sample AOI/Mask shapefiles and rasters
│   ├── outputdata/                  # Simulation Results output directory
│   └── scenarios/                   # Ready-to-use scenario YAML files
│
├── plugin/                       # QGIS plugin — launches this same desktop GUI inside QGIS
├── environment.yml               # Conda/Mamba Environment File (Recommended)
├── requirements.txt              # Pip Requirements File
├── pytest.ini                    # pytest configuration
└── README.md                     # Project Documentation
```

</details>

## 🛠️ Installation

Because this project relies on GDAL and Rasterio, we strictly recommend using Conda or Mamba to manage binary dependencies.

Option 1: Using Mamba (Recommended)
``` bash
# 1. Clone the repository
git clone https://github.com/ashutoshkumarjha/OpenLDMPy.git
cd OpenLDMPy

# 2. Create the environment
mamba env create -f environment.yml

# 3. Activate the environment
mamba activate openldm
```
Option 2: Using Pip
Warning: You must manually ensure GDAL C++ headers are installed on your system.

``` bash
pip install -r requirements.txt
```

> **Caution:** `environment.yml` pins `pyqt`, `pyqtwebengine`, and `qt-webengine` together as a matched conda-forge build. Running `pip install -r requirements.txt` *on top of* a conda env created from `environment.yml` will silently overwrite the conda-managed `pyqtwebengine` files with a mismatched pip build, crashing the GUI's Help tab (`QWebEngineView`) — this happened for real during development. Use one or the other, never both on the same environment.

## 🖥️ Usage

`src/OpenLDM.py` is the single entry point for both the GUI and headless runs.

1. Running the GUI (default mode)
Pure Python (PyQt5), no `rpy2`/R dependency.

``` bash
python src/OpenLDM.py
# equivalently: python src/OpenLDM.py --mode gui
```

Load a scenario (see below) into a running GUI via File > Open, or export one via File > Save.

2. Running headlessly (no GUI)
Ideal for testing algorithms or running on HPC clusters/CI. Requires a scenario YAML file — a human-readable "parameters file" describing input rasters, drivers, per-class settings, allocation/spatial-context options, and accuracy assessment (format defined in `LULC/scenario.py`, loaded via `ScenarioFile.to_run_config`). The GUI's File > Save writes one from the current session; `data/example/scenarios/runSteps1a.yaml` is a ready-to-use example mirroring the bundled `data/example/` dataset and the retired R backend's `runSteps1a.R` (scenario 5).

``` bash
python src/OpenLDM.py --mode nogui --config data/example/scenarios/runSteps1a.yaml
```

On successful execution, the predicted map is written to the `outputfile` path set in the scenario (plus `-Step1.tif`/`-Step2.tif` etc. for multi-step runs), alongside per-class `<className>SM.tif` (suitability) and `<className>NW.tif` (neighborhood weight) maps in the configured suitability-maps directory.

## 🧰 Development

### 🧪 Testing

Run tests as you develop, not just before a PR.

``` bash
pytest tests/
```

`tests/` mirrors `src/LULC/` and `src/gui/` file-for-file. Run the matching file for whatever you touch: `pytest tests/test_<module>.py`.

<details>
<summary>Click to expand test file list</summary>

- `test_OpenLDM.py`
- `test_transition.py`
- `test_allocation.py`
- `test_accuracy.py`
- `test_pontius.py`
- `test_masking.py`
- `test_modeling.py`
- `test_rasterize.py`
- `test_scenario.py`
- `test_gui_controller.py`
- `test_gui_smoke.py`
- `test_pipeline_integration.py`

</details>

Some tests check outputs byte-exact against R-backend fixtures (`tests/r_oracle/fixtures/`); these auto-skip until regenerated via `Rscript tests/r_oracle/generate_fixtures.R`. R is never required to run the app or the suite day-to-day.

### 🤝 Contributing

Pull requests are welcome. For major changes, open an issue first to discuss what you'd like to change, and add or update tests as appropriate (see [Testing](#-testing) above).

## 📊 Workflow & Logic

`LULCAlgorithms.py` is the facade (`run_pipeline`/`generate_predicted_map`) both `OpenLDM.py --mode nogui` and `gui/controller.py` call into, sitting atop a GUI → Controller → Processing → Data layering. The pipeline stages:

### 1. Data Ingestion & Setup (raster_io.py, masking.py, rasterize.py, scenario.py):
  - Reads T1 (Start Year) and T2 (End Year) LULC rasters and driver stacks as array-native `RasterLayer`s (not DataFrames).
  - Applies AOI/mask clipping (raster or shapefile, via `geopandas`).
  - Optionally converts a vector shapefile to a raster class layer first (`rasterize.rasterise`, wired to the Data Preparation tab's "Convert" buttons).
  - A run's full parameter set — inputs, drivers, per-class settings, allocation/spatial options — can be saved to and loaded from a YAML scenario file (`scenario.ScenarioFile`), shared by the GUI's File > Open/Save and the `--mode nogui --config` CLI path.

### 2. Transition Modeling (transition.py):

 - Calculates the Transition Matrix (Markov Chain) between T1 and T2.

 - Projects the future transition matrix (Markov or demand-constrained via iterative proportional fitting) and decomposes it into per-step matrices for multi-step simulation.

### 3. Suitability Prediction (modeling.py):

- Extracts training data (T1 Classes vs. T1 Drivers).

 - Trains binary classifiers (One-vs-Rest) for each land class — logistic, random forest, SVM, or MLP.

 - Predicts suitability probabilities using T2 Drivers.

 - Performance: Runs in parallel using all available CPU cores (joblib).

### 4. Spatial Context (spatial.py):
  - Computes per-class neighborhood/proximity weights (`scipy.ndimage` distance transforms) that feed into allocation as a spatial-inertia term.

### 5. Allocation (allocation.py):
  - Iteratively allocates pixels to meet land demand across a 4-phase algorithm (inertia retention, competitive allocation, optimum fallback, retreat pass) — a faithful port of the R backend's `getAllocatedDT`.
  - Resolves competition based on suitability scores, conversion order, spatial inertia, and neighborhood proximity.

### 6. Accuracy Assessment (accuracy.py, pontius.py):
  - Confusion matrix and Cohen's kappa statistics (overall + per-class), matching the R backend's `kappa`/`PyKappasummary` output format exactly.
  - Pontius agreement-index disagreement decomposition (`pontius.py`) is also implemented and tested against an R-oracle fixture, though not yet wired to a GUI button.

### 7. Visualization & Output (cartography.py):
  - Renders classified maps (legend, scale bar, north arrow) via matplotlib/rasterio, wired to the View Maps tab's "Show"/"Export".
  - Predicted maps, per-class suitability/neighborhood-weight maps, and session logs are written using the same file/naming conventions as the retired R backend.

Error handling throughout the pipeline uses a small custom exception hierarchy (`LULC.errors`: `PipelineError` → `DatasetValidationError`, `AllocationError`) so the CLI/GUI can show a clean message instead of a traceback for user-actionable failures.

## Citation

### Software

@misc{Jha2020, author = {Jha, Ashutosh Kumar }, title = {Open land use land cover Dynamic Modeling Platform (OpenLDMP)}, year = {2020}, publisher = {GitHub}, journal = {GitHub repository}, howpublished = {\url{https://github.com/ashutoshkumarjha/OpenLDM}}, commit = {29b942808ea9cd371fca4a0747e1e85452e02181} }

### Paper

@article{Jha2022, author = {Ashutosh Kumar Jha and S. K. Ghosh and S. K. Srivastav and Sameer Saran}, doi = {10.1007/s12524-022-01516-9}, issn = {0255-660X}, journal = {Journal of the Indian Society of Remote Sensing}, month = {2}, title = {OpenLDM: Open-Source Land-Use and Land-Cover Dynamics Modeling Platform}, url = { https://link.springer.com/10.1007/s12524-022-01516-9 }, year = {2022}, }

## 📝 License
[GPL-3.0](LICENSE)
