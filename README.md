# OpenLDM: Open-source Land-use Dynamics Modeling Platform

**OpenLDM** is a high-performance, Python-based geospatial modeling platform designed to simulate and predict Land Use and Land Cover (LULC) changes. It integrates advanced Machine Learning algorithms with spatial neighborhood constraints and competitive demand allocation logic to project future landscape scenarios.

> **Note:** This version marks a complete migration from the legacy R-based backend to a native Python architecture, utilizing `scikit-learn` for parallelized modeling and `rasterio` for efficient geospatial I/O.

---

## 🚀 Key Features

*   **Machine Learning Integration:** Train models using **Random Forest**, **SVM**, **Neural Networks (MLP)**, or **Logistic Regression** to predict land suitability based on driver variables (e.g., distance to roads, elevation, slope).
*   **High Performance:** Utilizes multi-core parallel processing (via `joblib`) for efficient model training and prediction on high-resolution rasters.
*   **Competitive Allocation Algorithm:** Implements an iterative, demand-driven allocation logic that handles complex competition between land classes based on suitability and transition matrices.
*   **Spatial Constraints:** Accounts for neighborhood effects, spatial inertia (persistence), and dynamic transition probabilities.
*   **Dual Interface:**
    *   **GUI:** User-friendly **PyQt5** interface for project setup, driver selection, and visual parameter configuration.
    *   **CLI:** Headless `main.py` script optimized for HPC environments or batch processing.

---

## 📂 Project Structure

```text
OpenLDMPy/
├── src/
│   ├── LULC/                   # Core Python Package
│   │   ├── __init__.py         # Package initialization
│   │   ├── config.py           # Global settings (Parallel jobs, Logging, Constants)
│   │   ├── data_io.py          # Raster/Driver I/O (Rasterio/Pandas)
│   │   ├── modeling.py         # ML Engine (Scikit-Learn/Joblib)
│   │   ├── spatial.py          # Neighborhood calculations & Markov Logic
│   │   ├── allocation.py       # Competitive Allocation Algorithm
│   │   └── LULCAlgorithms.py   # Main API Facade connecting Logic to GUI/CLI
│   │
│   ├── runLULCgui5.py          # GUI Launcher (PyQt5)
│   ├── LULCgui.py              # GUI Layout Definition
│   └── main.py                 # CLI/Headless Launcher for testing/automation
│
├── example/                    # Data Directory
│   ├── LULC/                   # Input Rasters (e.g., 1985.tif, 1995.tif)
│   ├── Drivers/                # Spatial Drivers (Slope, Road dist, etc.)
│   └── outputdata/             # Simulation Results output directory
│
├── environment.yml             # Conda/Mamba Environment File (Recommended)
├── requirements.txt            # Pip Requirements File
└── README.md                   # Project Documentation
```
## 🛠️ Installation

Because this project relies on GDAL and Rasterio, we strictly recommend using Conda or Mamba to manage binary dependencies.

Option 1: Using Mamba (Recommended)
``` bash
# 1. Clone the repository
git clone https://github.com/yourusername/OpenLDMPy.git
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
## 🖥️ Usage
1. Running the CLI (Headless Mode)
Ideal for testing algorithms or running on HPC clusters.

Edit src/main.py to point to your specific data paths (T1, T2, Drivers).

Run the script:

``` bash
python src/main.py
```

2. Running the GUI
A visual interface for setting up the model, selecting drivers, and defining policies.

``` bash
python src/runLULCgui5.py
```

## 📊 Workflow & Logic
### 1. Data Ingestion (data_io.py):
  - Reads T1 (Start Year) and T2 (End Year) LULC rasters.
  - Reads Driver variables (Distance to Roads, Elevation, etc.) and stacks them into DataFrames.

### 2. Transition Modeling (spatial.py):

 - Calculates the Transition Matrix (Markov Chain) between T1 and T2.

 - Decomposes multi-year changes into yearly steps using Matrix Logarithms (scipy.linalg.logm).

### 3. Suitability Prediction (modeling.py):

- Extracts training data (T1 Classes vs. T1 Drivers).

 - Trains binary classifiers (One-vs-Rest) for each land class.

 - Predicts suitability probabilities using T2 Drivers.

 - Performance: Runs in parallel using all available CPU cores.

### 4. Allocation (allocation.py):
  - Iteratively allocates pixels to meet land demand.
  - Resolves competition based on suitability scores, conversion elasticity, and spatial restrictions.


## 📝 License
[MIT License](https://www.perplexity.ai/search/LICENSE)