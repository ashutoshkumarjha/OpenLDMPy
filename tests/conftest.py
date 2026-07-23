import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
DATA = ROOT / "data" / "example"
FIXTURES = Path(__file__).resolve().parent / "r_oracle" / "fixtures"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# GUI tests need no real display; default to Qt's offscreen
# platform plugin unless the environment already requests one (e.g. a
# developer running with a real display wants normal windows).
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# QtWebEngine (used by the Help tab) spawns a Chromium renderer subprocess
# that HANGS INDEFINITELY constructing QWebEngineView in this sandboxed
# environment — confirmed even with QTWEBENGINE_DISABLE_SANDBOX=1 and
# --no-sandbox/--disable-gpu Chromium flags set at the OS-environment level
# (three independent hung runs, 0% CPU for 5-29 minutes each with a stuck
# QtWebEngineProcess child), so this is an IPC restriction from the outer
# sandbox itself, not Chromium's own sandboxing. Since there is no reliable
# flag-based fix, QWebEngineView is stubbed out entirely for tests below
# (production code — gui/OpenLDMgui.py, OpenLDM.py's run_gui() — is untouched;
# real desktop launches still get the genuine widget).
try:
    from PyQt5 import QtWebEngineWidgets, QtWidgets

    class _StubWebEngineView(QtWidgets.QWidget):
        """Drop-in stand-in for QWebEngineView: a real QWidget (so
        layout.addWidget(...) and setProperty/setObjectName work normally)
        plus the one QWebEngineView-specific method OpenLDMgui.py's setupUi and
        main_window.py's loadHelpFile actually call."""

        def setHtml(self, *args, **kwargs):
            pass

    QtWebEngineWidgets.QWebEngineView = _StubWebEngineView
except ImportError:
    pass


def has_fixture(name: str) -> bool:
    return (FIXTURES / name).exists()


def load_csv_matrix(name: str) -> np.ndarray:
    return np.loadtxt(FIXTURES / name, delimiter=",")


skip_if_missing_fixture = pytest.mark.skipif


@pytest.fixture(scope="session")
def data_dir():
    return DATA


@pytest.fixture(scope="session")
def fixtures_dir():
    return FIXTURES


@pytest.fixture(scope="session")
def class_names():
    return [
        "BuildUp", "Agriculture", "DenseForest", "FallowLand",
        "GrassLand", "MixedForest", "Plantation", "ScrubLand", "WaterBody",
    ]


@pytest.fixture(scope="session")
def t1_file(data_dir):
    return str(data_dir / "LULC/1985.tif")


@pytest.fixture(scope="session")
def t2_file(data_dir):
    return str(data_dir / "LULC/1995.tif")


@pytest.fixture(scope="session")
def t3_file(data_dir):
    return str(data_dir / "LULC/2005.tif")


@pytest.fixture(scope="session")
def drivers_85(data_dir):
    return {
        "DistanceToDrainage": str(data_dir / "Drivers/drivers_85/dist_stream.img"),
        "DistanceToBuiltup": str(data_dir / "Drivers/drivers_85/Dist_urban.img"),
        "DistanceToRoad": str(data_dir / "Drivers/drivers_85/road_final.img"),
        "Elevation": str(data_dir / "Drivers/commonDrivers/elevation.img"),
    }


@pytest.fixture(scope="session")
def drivers_95(data_dir):
    return {
        "DistanceToDrainage": str(data_dir / "Drivers/drivers_95/dist_stream.img"),
        "DistanceToBuiltup": str(data_dir / "Drivers/drivers_95/Dist_urban.img"),
        "DistanceToRoad": str(data_dir / "Drivers/drivers_95/road_final.img"),
        "Elevation": str(data_dir / "Drivers/commonDrivers/elevation.img"),
    }
