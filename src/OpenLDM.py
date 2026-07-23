"""Unified entry point for OpenLDM: GUI or headless pipeline runs.

    python OpenLDM.py                                     # launch the GUI (default mode)
    python OpenLDM.py --mode gui --config <path>           # launch the GUI, pre-loaded with a scenario
    python OpenLDM.py --mode nogui --config <path>         # run a scenario headlessly, no GUI

``--config`` is a scenario YAML file (see ``LULC/scenario.py``) — the same
format the GUI's File > Open/Save menu actions read and write. It is
**required** when ``--mode nogui`` is given (there is no headless default
scenario). In ``gui`` mode it's optional: if given, it's loaded into the
window on startup exactly as File > Open would (every tab populated, no
compute step run); omit it to start with an empty session and load one
interactively later. Any field a scenario file leaves unset gets a
default the same way an unchecked GUI checkbox would (see
``ScenarioFile.to_run_config``/``LULCAlgorithms.build_run_config``).

A missing, corrupted, or malformed ``--config`` file never crashes either
mode: ``gui`` mode still opens the window and reports the problem in a
message box; ``nogui`` mode prints a plain-text error to stderr and exits
with status 1, no raw traceback.

Replaces the former split between this file (headless-only) and
``runLULCgui5.py`` (GUI-only, now merged in here) — one entry point for
both.
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))


def run_gui(config_path: Optional[str] = None) -> int:
    """Launch the PyQt5 desktop GUI. Imports are local to this function so
    ``--mode nogui`` usage never needs PyQt5/Qt importable at all (a
    headless/server environment without a display can run pipelines
    without any GUI dependency being present).

    ``config_path``, if given, is loaded into the running window exactly
    as File > Open would (``MyForm.loadScenarioFile`` — same method, same
    behavior): every tab gets populated, no compute step runs. A
    malformed/corrupted file does **not** prevent the GUI from opening —
    it still shows, with a QMessageBox reporting what's wrong."""
    # QWebEngineView (the Help tab) used to segfault deep inside
    # libQt5WebEngineCore, on the Chrome_InProcGpuThread, the moment it was
    # actually painted — root-caused via a macOS crash report to a
    # null-pointer dereference in Chromium's GPU/compositor thread. Qt5
    # WebEngine is EOL and bundles a Chromium build frozen years ago; its
    # GPU-process graphics-driver interop code was never tested against
    # this host's macOS version, which is far newer. Disabling Chromium's
    # GPU process (forcing software compositing) routes around it entirely
    # — confirmed via a dozen repro runs both with and without this flag,
    # and independently of process sandboxing (which fixes an unrelated
    # *hang*, not this crash, in the headless test environment — see
    # tests/conftest.py). Must be set before QtWebEngineWidgets is
    # imported: Chromium reads its flags at that module's own init time,
    # not at QApplication construction.
    os.environ.setdefault("QTWEBENGINE_CHROMIUM_FLAGS", "--disable-gpu")

    from PyQt5 import QtCore, QtGui, QtWidgets
    from PyQt5 import QtWebEngineWidgets  # noqa: F401 - must import before QApplication (Qt quirk)

    from gui.main_window import LULCMainWindow

    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)
    app = QtWidgets.QApplication(sys.argv)
    window = LULCMainWindow()
    window.show()
    window.setWindowIcon(QtGui.QIcon(":/images/images/icon.png"))
    if config_path:
        window.loadScenarioFile(config_path)
    return app.exec_()


def run_nogui(config_path: str) -> int:
    """Run a scenario headlessly, no GUI at all. Returns a process exit
    code: 0 on success, 1 if the scenario file couldn't be loaded (missing,
    corrupted, malformed YAML, or missing required fields) -- reported as
    plain text on stderr rather than a raw traceback, the nogui-mode
    equivalent of the GUI's QMessageBox.critical on the same failures."""
    from LULC import LULCAlgorithms as Algorithm
    from LULC.scenario import ScenarioFile, ScenarioFileError

    print(f"Loading scenario: {config_path}")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    try:
        scenario = ScenarioFile.from_yaml(config_path)
        cfg = scenario.to_run_config()
    except ScenarioFileError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    print("\n--- Running prediction pipeline ---")
    result = Algorithm.run_pipeline(cfg)
    print(f"\n--- Prediction Complete: {result.output_file} ---")

    acc = scenario.accuracy_assessment
    reference_file = acc.reference_file
    # Same auto-sync the GUI's Output File field applies to "Predicted
    # File" (main_window.py's on_leOutputFile_..._editingFinished) --
    # explicit acc.predicted_file overrides it, matching the GUI's own
    # "independently selectable" behavior.
    predicted_file = acc.predicted_file or result.output_file

    if reference_file and os.path.exists(reference_file) and os.path.exists(predicted_file):
        print("\n--- Calculating Accuracy (Kappa) ---")
        summary = Algorithm.get_kappa_summary(
            actualFile=reference_file,
            predictedFile=predicted_file,
            na_value=scenario.na_value,
            classNames=cfg.class_names,
        )
        print(f"Overall accuracy: {summary['accuracy']:.4f}")
        print(f"Overall kappa:    {summary['kappa']:.4f}")

        if acc.base_file and os.path.exists(acc.base_file):
            print("\n--- Pontius Agreement Index (base file provided) ---")
            pontius_result = Algorithm.kappa_agreement_index(
                predicted_file, reference_file, acc.base_file, na_value=scenario.na_value,
            )
            print(pontius_result.summary_text(cfg.class_names))
        elif acc.base_file:
            print(f"\n(Base file {acc.base_file} not found -- skipping Pontius agreement index)")

    end_time = datetime.now()
    print(f"\nEnd Time: {end_time}")
    print(f"Total Duration: {end_time - start_time}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=["gui", "nogui"], default="gui",
        help="'gui' (default): launch the desktop app. 'nogui': run a "
             "scenario headlessly (requires --config).",
    )
    parser.add_argument(
        "--config", metavar="PATH",
        help="Path to a scenario YAML file (see LULC/scenario.py). "
             "Required with --mode nogui. In gui mode, loads it into the "
             "window on startup (same as File > Open).",
    )
    args = parser.parse_args()

    if args.mode == "nogui":
        if not args.config:
            parser.error("--config is required with --mode nogui")
        return run_nogui(args.config)

    return run_gui(args.config)


if __name__ == "__main__":
    sys.exit(main())
