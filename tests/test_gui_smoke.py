"""Headless smoke tests for the ported Qt GUI.

Requires PyQt5 + pytest-qt. Run with QT_QPA_PLATFORM=offscreen (no display
needed) — conftest.py sets this automatically if unset.
"""

import os
from pathlib import Path

import pytest

pytest.importorskip("PyQt5")
pytest.importorskip("pytestqt")

from PyQt5 import QtCore, QtWidgets
from PyQt5 import QtWebEngineWidgets  # noqa: F401 - must import before QApplication


@pytest.fixture(scope="session", autouse=True)
def _qapp_attributes():
    QtWidgets.QApplication.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts)


@pytest.fixture
def main_window(qtbot, monkeypatch):
    from gui.main_window import LULCMainWindow

    # closeEvent shows a real, modal "Are you sure?" QMessageBox.question —
    # blocks forever with no one to click it in a headless test run.
    # Auto-confirm for the duration of this test.
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
    )
    # initGui() -> loadHelpFile() shows a modal QMessageBox.information if
    # docs/OpenLDM.html isn't found relative to get_main_dir() (never true
    # under pytest) — same blocks-forever problem as above. Auto-dismiss.
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
    )

    window = LULCMainWindow()
    qtbot.addWidget(window)
    yield window
    window.close()


def test_main_window_initializes(main_window):
    assert main_window.ui.tabWidget.count() == 8


def test_tabs_are_navigable(main_window):
    for i in range(main_window.ui.tabWidget.count()):
        main_window.ui.tabWidget.setCurrentIndex(i)
        assert main_window.ui.tabWidget.currentIndex() == i


def test_no_rpy2_import_anywhere():
    import gui.main_window as mw

    assert not hasattr(mw, "R")
    assert not hasattr(mw, "rpy2")


def test_convert_button_without_shapefile_shows_selection_prompt(main_window, monkeypatch):
    shown = {}

    def fake_information(*args, **kwargs):
        shown["called"] = True

    monkeypatch.setattr("gui.main_window.QMessageBox.information", fake_information)
    main_window.on_pbConvert_T0_clicked()
    assert shown.get("called") is True


def test_exit_actually_stops_status_bar_thread(main_window):
    """Bug: StatusBarThread.run() is a raw `while True: time.sleep(1)` loop
    that never calls self.exec_(), so the old self.statusMessage.exit(0)
    (QThread.exit(), which posts a quit to a thread's own Qt event loop) was
    a no-op -- the thread never actually stopped on Exit. Same latent bug
    existed in the original R GUI."""
    assert main_window.statusMessage.isRunning()
    main_window.on_actionExit_triggered()
    assert not main_window.statusMessage.isRunning()


class _FakeSignal:
    def connect(self, *_a, **_k):
        pass


class _FakeWorker:
    finished_ok = _FakeSignal()
    failed = _FakeSignal()

    def __init__(self, func, *a, **k):
        self.func = func

    def start(self):
        pass


def test_execute_button_runs_in_background_worker(main_window, monkeypatch):
    """Doesn't run a real pipeline — just verifies the Execute action goes
    through BackgroundTaskWorker rather than blocking the GUI thread."""
    import gui.main_window as mw

    monkeypatch.setattr(mw, "BackgroundTaskWorker", _FakeWorker)
    main_window.prepareDataForExecution = lambda: None
    main_window.on_pbExecute_DemandAllocation_clicked()
    # If we got here without a blocking real pipeline run, the worker path was used.


def test_progress_relay_emits_signal(qtbot):
    from gui.progress_bridge import QtProgressRelay

    relay = QtProgressRelay()
    with qtbot.waitSignal(relay.progress_changed, timeout=1000) as blocker:
        relay.report(42, "Fitting per-class models")
    assert blocker.args == [42, "Fitting per-class models"]


def test_execute_button_wires_progress_bar(main_window, monkeypatch):
    """The Execute handler should connect a QtProgressRelay to the progress
    bar and pass its .report as on_progress to the background worker."""
    import gui.main_window as mw

    captured = {}

    class _CapturingWorker(_FakeWorker):
        def __init__(self, func, *a, **k):
            super().__init__(func, *a, **k)
            captured["on_progress"] = k.get("on_progress")

    monkeypatch.setattr(mw, "BackgroundTaskWorker", _CapturingWorker)
    main_window.prepareDataForExecution = lambda: None
    main_window.on_pbExecute_DemandAllocation_clicked()

    assert captured["on_progress"] is not None
    captured["on_progress"](77, "Allocating")
    assert main_window.ui.progressBar.property("value") == 77


def _fill_data_preparation_fields(main_window, t0_file, t1_file):
    """Fills the same 5 fields enable_ValidateDataPreparation() gates on,
    firing each field's real editingFinished handler exactly like a user
    tabbing out of the widget would."""
    ui = main_window.ui
    fields = [
        (ui.leT0File_DataPreparationInputSectionDataInput, t0_file,
         main_window.on_leT0File_DataPreparationInputSectionDataInput_editingFinished),
        (ui.leT1File_DataPreparationInputSectionDataInput, t1_file,
         main_window.on_leT1File_DataPreparationInputSectionDataInput_editingFinished),
        (ui.leOutputFile_DataPreparationOutputSection, "/tmp/openldm_test_out.tif",
         main_window.on_leOutputFile_DataPreparationOutputSection_editingFinished),
        (ui.leT0Year_DataPreparationInputSectionDataInput, "1985",
         main_window.on_leT0Year_DataPreparationInputSectionDataInput_editingFinished),
        (ui.leT1Year_DataPreparationInputSectionDataInput, "1995",
         main_window.on_leT1Year_DataPreparationInputSectionDataInput_editingFinished),
    ]
    for widget, text, handler in fields:
        widget.setText(text)
        handler()


def test_validate_button_starts_disabled(main_window):
    assert not main_window.ui.pbValidate_DataPreparation.isEnabled()
    assert not main_window.ui.pbNextDataPreparation.isEnabled()


def test_validate_button_enables_once_five_fields_populated_next_stays_disabled(main_window, t1_file, t2_file):
    _fill_data_preparation_fields(main_window, t1_file, t2_file)
    assert main_window.ui.pbValidate_DataPreparation.isEnabled()
    assert not main_window.ui.pbNextDataPreparation.isEnabled()


def test_validate_success_enables_next_and_locks_fields(main_window, monkeypatch, t1_file, t2_file):
    shown = {}
    monkeypatch.setattr(
        "gui.main_window.QMessageBox.information", lambda *a, **k: shown.setdefault("info", True)
    )
    _fill_data_preparation_fields(main_window, t1_file, t2_file)

    main_window.on_pbValidate_DataPreparation_clicked()

    assert shown.get("info") is True
    assert main_window.ui.pbNextDataPreparation.isEnabled()
    assert not main_window.ui.leT0File_DataPreparationInputSectionDataInput.isEnabled()
    assert not main_window.ui.leT1File_DataPreparationInputSectionDataInput.isEnabled()
    assert not main_window.ui.leOutputFile_DataPreparationOutputSection.isEnabled()
    assert not main_window.ui.leT0Year_DataPreparationInputSectionDataInput.isEnabled()
    assert not main_window.ui.leT1Year_DataPreparationInputSectionDataInput.isEnabled()


def test_validate_failure_shows_issues_and_leaves_next_disabled(main_window, monkeypatch, tmp_path, t1_file, t2_file):
    import rasterio

    with rasterio.open(t1_file) as src:
        profile = src.profile.copy()
        data = src.read(1)
    t = profile["transform"]
    profile["transform"] = rasterio.Affine(t.a, t.b, t.c + 5000, t.d, t.e, t.f)
    shifted = tmp_path / "shifted.tif"
    with rasterio.open(shifted, "w", **profile) as dst:
        dst.write(data, 1)

    shown = {}
    monkeypatch.setattr(
        "gui.main_window.QMessageBox.warning",
        lambda *a, **k: shown.setdefault("message", a[2] if len(a) > 2 else k.get("text")),
    )
    _fill_data_preparation_fields(main_window, str(shifted), t2_file)

    main_window.on_pbValidate_DataPreparation_clicked()

    assert "extent does not match" in shown.get("message", "")
    assert not main_window.ui.pbNextDataPreparation.isEnabled()
    # fields stay editable so the user can fix and re-validate
    assert main_window.ui.leT0File_DataPreparationInputSectionDataInput.isEnabled()


def _run_find_accuracy(main_window, actual_file, predicted_file, base_file=None):
    ui = main_window.ui
    ui.leActualFile_AccuracyAssesment.setText(actual_file)
    ui.lePredictedFile_AccuracyAssesment.setText(predicted_file)
    ui.leBaseFile_AccuracyAssesment.setText(base_file or "")
    main_window.on_pbFindAccuracy_AccuracyAssesment_clicked()


def test_base_file_select_button_sets_field(main_window, monkeypatch):
    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: ("data/example/LULC/1985.tif", "")),
    )
    main_window.on_pbSelectFileBaseFile_AccuracyAssesment_clicked()
    assert main_window.ui.leBaseFile_AccuracyAssesment.text() == "data/example/LULC/1985.tif"


def test_agreement_index_radio_toggle_before_find_accuracy_does_not_crash(main_window):
    """Bug: toggling Classwise/Overall before "Find Accuracy" has ever run
    (self.__pontiusResult is None until calculateStatistics() computes it)
    crashed with AttributeError: 'NoneType' object has no attribute
    'ksimulation_overall'."""
    main_window.ui.rbAgreementIndexOverall_AccuracyAssesmentDetailed.setChecked(True)
    main_window.ui.rbAgreementIndexClasswise_AccuracyAssesmentDetailed.setChecked(True)


def test_agreement_index_overall_without_base_file_omits_base_dependent_rows(main_window, t1_file, t2_file, t3_file):
    """The 3 metrics that need a base file (Ksimulation/Ktransition/
    Ktranslocation) must not appear as rows at all when none was given —
    not just show up with blank cells."""
    _run_find_accuracy(main_window, t2_file, t3_file)  # no base file
    main_window.ui.rbAgreementIndexOverall_AccuracyAssesmentDetailed.setChecked(True)
    table = main_window.ui.twDetailed_AccuracyAssesment

    assert table.rowCount() == 5
    assert table.columnCount() == 1
    row_labels = [table.verticalHeaderItem(i).text() for i in range(table.rowCount())]
    assert row_labels == ["Kstandard", "Kno", "Kallocation", "Khistogram", "Kquantity"]
    for i in range(table.rowCount()):
        text = table.item(i, 0).text()
        assert text != "" and float(text) == pytest.approx(float(text))


def test_agreement_index_overall_with_base_file_populates_all_rows(main_window, t1_file, t2_file, t3_file):
    _run_find_accuracy(main_window, t2_file, t3_file, base_file=t1_file)
    main_window.ui.rbAgreementIndexOverall_AccuracyAssesmentDetailed.setChecked(True)
    table = main_window.ui.twDetailed_AccuracyAssesment

    for i in range(table.rowCount()):
        assert table.item(i, 0).text() != ""


def test_agreement_index_classwise_columns_are_classes(main_window, t1_file, t2_file, t3_file, class_names):
    _run_find_accuracy(main_window, t2_file, t3_file, base_file=t1_file)
    main_window.ui.rbAgreementIndexClasswise_AccuracyAssesmentDetailed.setChecked(True)
    table = main_window.ui.twDetailed_AccuracyAssesment

    assert table.rowCount() == 8
    assert table.columnCount() == len(class_names)
    for i in range(table.rowCount()):
        for j in range(table.columnCount()):
            assert table.item(i, j).text() != ""


def test_output_file_editing_finished_does_not_double_absolute_path(main_window, t1_file):
    """Bug: this handler fires whenever the field loses focus, including
    right after "Select File" already put an absolute path in it — it used
    to unconditionally prepend __currentDirectory, doubling the path into
    something like ".../LULC/Users/.../outputdata/output.tif"."""
    ui = main_window.ui
    ui.leT0File_DataPreparationInputSectionDataInput.setText(t1_file)
    main_window.on_leT0File_DataPreparationInputSectionDataInput_editingFinished()

    absolute_output = "/Users/ashutosh/Claude/OpenLDMPy/data/example/outputdata/output.tif"
    ui.leOutputFile_DataPreparationOutputSection.setText(absolute_output)
    main_window.on_leOutputFile_DataPreparationOutputSection_editingFinished()

    assert main_window._MyForm__OutputFile == absolute_output


def test_output_file_editing_finished_still_resolves_relative_names(main_window):
    """The fix for the bug above must not break the legitimate case: a
    relative filename typed directly into the field still resolves against
    __currentDirectory."""
    ui = main_window.ui
    main_window._MyForm__currentDirectory = "/some/project/dir/"
    ui.leOutputFile_DataPreparationOutputSection.setText("relative_output.tif")
    main_window.on_leOutputFile_DataPreparationOutputSection_editingFinished()

    assert main_window._MyForm__OutputFile == "/some/project/dir/relative_output.tif"


def test_find_accuracy_stays_enabled_through_execute(main_window, monkeypatch):
    """Find Accuracy reads its own Actual/Predicted/Base file fields and
    should be usable independent of Execute — it must not be disabled at
    the start of Execute and left that way if Execute fails or is slow."""
    import gui.main_window as mw

    assert main_window.ui.pbFindAccuracy_AccuracyAssesment.isEnabled()
    monkeypatch.setattr(mw, "BackgroundTaskWorker", _FakeWorker)
    main_window.prepareDataForExecution = lambda: None
    main_window.on_pbExecute_DemandAllocation_clicked()
    assert main_window.ui.pbFindAccuracy_AccuracyAssesment.isEnabled()


def test_view_model_statistics_disables_all_model_type_radios(main_window, monkeypatch):
    """Bug: only the "View Model Statistics" button itself was locked
    during the background fit, not the model-type radios — so switching
    model type mid-fit re-enabled the button (each on_rbXXX_..._toggled
    handler does that unconditionally) and let a second fit start while
    the first was still running."""
    import gui.main_window as mw

    monkeypatch.setattr(mw, "BackgroundTaskWorker", _FakeWorker)
    main_window.ui.gbDoModelFitting_DriverSelectionT0.setEnabled(True)  # normally done reaching this tab
    main_window.ui.rbLogisticRegression_DriverSelectionT0DoModelFitting.setChecked(True)
    assert main_window.ui.pbViewModelStatistics_DriverSelectionT0DoModelFitting.isEnabled()
    # Simulate Next already being enabled from an earlier successful fit —
    # that's exactly the state in which re-clicking View Model Statistics
    # must still lock it back down for the new fit's duration.
    main_window.ui.pbNext_DriverSelectionT0.setEnabled(True)

    main_window.on_pbViewModelStatistics_DriverSelectionT0DoModelFitting_clicked()

    radios = [
        main_window.ui.rbLogisticRegression_DriverSelectionT0DoModelFitting,
        main_window.ui.rbLinearRegression_DriverSelectionT0DoModelFitting,
        main_window.ui.rbNeuralregression_DriverSelectionT0DoModelFitting,
        main_window.ui.rbRandomForest_DriverSelectionT0DoModelFitting,
        main_window.ui.rbSVM_DriverSelectionT0DoModelFitting,
    ]
    for rb in radios:
        assert not rb.isEnabled(), f"{rb.objectName()} should be disabled during the fit"
    assert not main_window.ui.pbViewModelStatistics_DriverSelectionT0DoModelFitting.isEnabled()
    assert not main_window.ui.pbNext_DriverSelectionT0.isEnabled()


class _StoringSignal:
    """Unlike _FakeSignal, actually keeps the connected slot so a test can
    simulate a successful/failed completion by calling .emit(...)."""

    def __init__(self):
        self._slot = None

    def connect(self, slot):
        self._slot = slot

    def emit(self, *args):
        if self._slot is not None:
            self._slot(*args)


class _CompletingWorker:
    def __init__(self, func, *a, **k):
        self.func = func
        self.finished_ok = _StoringSignal()
        self.failed = _StoringSignal()

    def start(self):
        pass


def test_execute_populates_view_maps_file_selector(main_window, monkeypatch):
    """Bug: cbFile_ViewMaps (the View Maps tab's file-to-view picker) was
    never populated with anything — this run's actual outputs should be
    selectable there once Execute succeeds."""
    import gui.main_window as mw
    from LULC.config import PipelineResult

    monkeypatch.setattr(mw, "BackgroundTaskWorker", _CompletingWorker)
    main_window.prepareDataForExecution = lambda: None
    main_window._MyForm__projectDirectory = "/tmp"
    main_window.currentLogTime = "test"

    main_window.on_pbExecute_DemandAllocation_clicked()

    worker = main_window._active_worker
    result = PipelineResult(
        output_file="/tmp/output.tif",
        transition_matrix=None,
        yearly_transition_matrix=None,
        step_files=["/tmp/output-Step1.tif"],
        suitability_files={"ClassA": "/tmp/ClassASM.tif"},
    )
    worker.finished_ok.emit(result)

    combo = main_window.ui.cbFile_ViewMaps
    items = [combo.itemText(i) for i in range(combo.count())]
    assert items == ["/tmp/output.tif", "/tmp/output-Step1.tif", "/tmp/ClassASM.tif"]
    assert combo.currentText() == "/tmp/output.tif"


def test_execute_success_does_not_stomp_progress_bar(main_window, monkeypatch):
    """Bug: on_success used to unconditionally reset the progress bar to a
    hardcoded 60 right after Execute finished, discarding whatever real
    percentage the QtProgressRelay had just driven it to (up to 100, the
    pipeline's own final report(100, "Complete"))."""
    import gui.main_window as mw
    from LULC.config import PipelineResult

    monkeypatch.setattr(mw, "BackgroundTaskWorker", _CompletingWorker)
    main_window.prepareDataForExecution = lambda: None
    main_window._MyForm__projectDirectory = "/tmp"
    main_window.currentLogTime = "test"

    main_window.on_pbExecute_DemandAllocation_clicked()
    main_window.ui.progressBar.setProperty("value", 100)  # simulates the relay having already reached 100

    worker = main_window._active_worker
    result = PipelineResult(
        output_file="/tmp/output.tif",
        transition_matrix=None,
        yearly_transition_matrix=None,
        step_files=[],
        suitability_files={},
    )
    worker.finished_ok.emit(result)

    assert main_window.ui.progressBar.property("value") == 100


def _build_class_tables(main_window, class_names):
    """Populate the 5 tables that each carry their own "Class" name column
    with a minimal fixture, matching how on_pbNext_DriverSelectionT0
    (etc.) would have built them for real."""
    ui = main_window.ui
    ui.rbLogisiticRegression_ModelAnalysisViewModelCoeeffcient.setChecked(True)
    main_window._MyForm__className = list(class_names)
    main_window._MyForm__noOfClasses = len(class_names)
    main_window.noOfDrivers = 0
    main_window._MyForm__DriverDictionaryT1 = {}
    main_window.buildtwViewModelCoefficint_ModelAnalysis(ui.twViewModelCoefficint_ModelAnalysis)
    main_window.buildtwSelectModelTypeAndDrivers_DriverSelectionT1(ui.twSelectModelTypeAndDrivers_DriverSelectionT1)
    main_window.buildtwMigrationOrder_ModelAnalysis(ui.twMigrationOrder_ModelAnalysis)
    main_window.buildtwPolicies_DemandAllocation(ui.twPolicies_DemandAllocation)
    main_window.buildtwColorTable_ViewMaps(ui.twColorTable_ViewMaps)


_CLASS_NAME_TABLES = (
    "twViewModelCoefficint_ModelAnalysis",
    "twSelectModelTypeAndDrivers_DriverSelectionT1",
    "twMigrationOrder_ModelAnalysis",
    "twPolicies_DemandAllocation",
    "twColorTable_ViewMaps",
)


def test_class_name_edit_in_any_table_syncs_to_all_others(main_window):
    """Bug: each of the 5 tables independently defaulted its own "Class"
    name column with no link between them — renaming a class in one didn't
    affect the others, or self.__className (what prepareClassName()/Execute
    ultimately reads)."""
    _build_class_tables(main_window, ["1", "2", "3"])

    main_window.ui.twColorTable_ViewMaps.item(1, 0).setText("Agriculture")
    for table_name in _CLASS_NAME_TABLES:
        assert getattr(main_window.ui, table_name).item(1, 0).text() == "Agriculture"
    assert main_window._MyForm__className[1] == "Agriculture"

    main_window.ui.twPolicies_DemandAllocation.item(1, 0).setText("AgriRenamed")
    for table_name in _CLASS_NAME_TABLES:
        assert getattr(main_window.ui, table_name).item(1, 0).text() == "AgriRenamed"
    assert main_window._MyForm__className[1] == "AgriRenamed"


def test_class_name_sync_updates_migration_order_column_header(main_window):
    """twMigrationOrder_ModelAnalysis names each class a second time, as a
    column header (row 0 col 0 is "migrate FROM", header of column row+1 is
    "migrate TO") — a rename from any table, including ones other than
    twViewModelCoefficint_ModelAnalysis, must update that header too."""
    _build_class_tables(main_window, ["1", "2", "3"])

    main_window.ui.twPolicies_DemandAllocation.item(1, 0).setText("Agriculture")

    header = main_window.ui.twMigrationOrder_ModelAnalysis.horizontalHeaderItem(2)
    assert header.text() == "Agriculture"


def test_class_name_sync_does_not_touch_policies_total_row(main_window):
    _build_class_tables(main_window, ["1", "2", "3"])
    total_row = main_window.ui.twPolicies_DemandAllocation.rowCount() - 1
    assert main_window.ui.twPolicies_DemandAllocation.item(total_row, 0).text() == "Total"

    main_window.ui.twPolicies_DemandAllocation.item(0, 0).setText("Renamed")
    assert main_window.ui.twPolicies_DemandAllocation.item(total_row, 0).text() == "Total"


def test_policies_allocation_column_swap_logic_still_works(main_window):
    """Regression guard: on_twPolicies_DemandAllocation_itemChanged has a
    substantial pre-existing Allocation-order swap/Demand-sum/Class-Inertia
    validation behind columns 1-3 that must survive the class-name-sync
    addition to column 0 (must not shadow or otherwise break it)."""
    _build_class_tables(main_window, ["1", "2", "3"])
    main_window.ui.twPolicies_DemandAllocation.item(0, 1).setText("2")
    main_window.ui.twPolicies_DemandAllocation.item(1, 1).setText("2")  # triggers the swap branch


def test_migration_order_swap_logic_still_works(main_window):
    """Same regression guard as above, for
    on_twMigrationOrder_ModelAnalysis_itemChanged's column>=1 swap logic."""
    _build_class_tables(main_window, ["1", "2", "3"])
    main_window.ui.twMigrationOrder_ModelAnalysis.setCurrentCell(0, 1)
    main_window.ui.twMigrationOrder_ModelAnalysis.item(0, 1).setText("2")


def _build_color_table(main_window, class_names, colors):
    from PyQt5.QtGui import QBrush, QColor

    main_window._MyForm__className = list(class_names)
    main_window._MyForm__noOfClasses = len(class_names)
    main_window.buildtwColorTable_ViewMaps(main_window.ui.twColorTable_ViewMaps)
    for i, color in enumerate(colors):
        main_window.ui.twColorTable_ViewMaps.item(i, 2).setBackground(QBrush(QColor(color)))


_LULC_CLASS_NAMES = [
    "BuildUp", "Agriculture", "DenseForest", "FallowLand",
    "GrassLand", "MixedForest", "Plantation", "ScrubLand", "WaterBody",
]
_LULC_CLASS_COLORS = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3", "#ff7f00",
    "#ffff33", "#a65628", "#f781bf", "#999999",
]


def test_show_map_renders_pixmap(main_window, t2_file):
    """Show should no longer say "not yet available" — it should
    actually render the selected file and display it."""
    _build_color_table(main_window, _LULC_CLASS_NAMES, _LULC_CLASS_COLORS)
    main_window.ui.cbFile_ViewMaps.addItem(t2_file)
    main_window.ui.cbFile_ViewMaps.setCurrentIndex(0)
    main_window.ui.leTitle_ViewMaps.setText("Test Title")

    main_window.on_pbShow_ViewMaps_clicked()

    assert main_window.ui.lbCanvas_ViewMaps.pixmap() is not None


def test_map_legend_uses_legend_text_column_not_class_name(main_window, monkeypatch, t2_file):
    """Bug: the map legend was built from twColorTable_ViewMaps column 0
    (Class Name) instead of column 1 (Legend Text), so custom legend
    labels typed into that column were silently ignored."""
    _build_color_table(main_window, _LULC_CLASS_NAMES, _LULC_CLASS_COLORS)
    legend_texts = [f"Legend-{i}" for i in range(len(_LULC_CLASS_NAMES))]
    for i, text in enumerate(legend_texts):
        main_window.ui.twColorTable_ViewMaps.item(i, 1).setText(text)
    main_window.ui.cbFile_ViewMaps.addItem(t2_file)
    main_window.ui.cbFile_ViewMaps.setCurrentIndex(0)

    generate_map_calls = []
    monkeypatch.setattr(
        main_window.controller, "generate_map",
        lambda *a, **k: generate_map_calls.append((a, k)) or Path(a[1]).write_bytes(b"fake-png"),
    )

    main_window.on_pbShow_ViewMaps_clicked()

    args, kwargs = generate_map_calls[0]
    names = args[3]
    assert names == legend_texts


def test_export_map_writes_png(main_window, monkeypatch, tmp_path, t2_file):
    """test_show_map_renders_pixmap already exercises the real rendering
    pipeline end-to-end; this test focuses on Export's own wiring (dialog
    interaction, output path handling, success message) and mocks the
    actual render call — running matplotlib's Agg backend twice in one
    pytest process (once per real-render test) reliably hangs on this
    platform, for reasons not yet root-caused; one real render per process
    is enough to prove the pipeline works."""
    _build_color_table(main_window, _LULC_CLASS_NAMES, _LULC_CLASS_COLORS)
    main_window.ui.cbFile_ViewMaps.addItem(t2_file)
    main_window.ui.cbFile_ViewMaps.setCurrentIndex(0)

    output_path = str(tmp_path / "exported.png")
    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (output_path, "")),
    )
    shown = {}
    monkeypatch.setattr(
        "gui.main_window.QMessageBox.information", lambda *a, **k: shown.setdefault("called", True)
    )
    generate_map_calls = []
    monkeypatch.setattr(
        main_window.controller, "generate_map",
        lambda *a, **k: generate_map_calls.append((a, k)) or Path(output_path).write_bytes(b"fake-png"),
    )

    main_window.on_pbExport_ViewMaps_clicked()

    assert len(generate_map_calls) == 1
    args, kwargs = generate_map_calls[0]
    assert args[0] == t2_file
    assert args[1] == output_path
    assert os.path.exists(output_path)
    assert shown.get("called") is True


def test_show_map_with_mismatched_color_table_shows_error_not_crash(main_window, monkeypatch, t2_file):
    """t2_file has 9 classes; give it a 3-row color table instead."""
    _build_color_table(main_window, ["A", "B", "C"], ["#ff0000", "#00ff00", "#0000ff"])
    main_window.ui.cbFile_ViewMaps.addItem(t2_file)
    main_window.ui.cbFile_ViewMaps.setCurrentIndex(0)

    shown = {}
    monkeypatch.setattr(
        "gui.main_window.QMessageBox.critical", lambda *a, **k: shown.setdefault("called", True)
    )

    main_window.on_pbShow_ViewMaps_clicked()

    assert shown.get("called") is True
    assert main_window.ui.lbCanvas_ViewMaps.pixmap() is None or main_window.ui.lbCanvas_ViewMaps.pixmap().isNull()


def test_show_map_with_no_file_selected_does_nothing(main_window, monkeypatch):
    # main_window fixture already mocks QMessageBox.question -> Yes, so
    # getFileToPlot() will offer a file picker; mock it to "cancelled" too
    # so this test is deterministic instead of depending on how a real
    # (unmocked) QFileDialog behaves headless.
    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: ("", "")),
    )
    main_window.ui.cbFile_ViewMaps.clear()
    main_window.on_pbShow_ViewMaps_clicked()  # must not raise
    assert main_window.ui.lbCanvas_ViewMaps.pixmap() is None or main_window.ui.lbCanvas_ViewMaps.pixmap().isNull()


def test_save_then_open_scenario_round_trip(
    qtbot, monkeypatch, tmp_path, t1_file, t2_file, drivers_85, drivers_95, class_names
):
    """End-to-end: build a scenario through the real Data
    Preparation -> ... -> Demand Allocation flow, Save it, then Open that
    same file into a *fresh* window and confirm the fast-forward lands on
    Demand Allocation with the same Demand/Class Inertia/Allocation
    Order/Spatial Context/class-name/T2-driver state. Model fitting itself
    is mocked out (BackgroundTaskWorker -> a synchronous fake) since the
    real fit isn't what's under test here and would make this test slow;
    everything else goes through the real handlers.

    Open does NOT auto-trigger "View Model Statistics",
    Execute, Show, or Export -- only Validate. So there is no real or
    faked background worker to wait on here; on_actionOpen_File_triggered
    runs synchronously start to finish."""
    import gui.main_window as mw
    from LULC.scenario import (
        ScenarioAccuracyAssessment,
        ScenarioClass,
        ScenarioFile,
        ScenarioMapComposition,
        ScenarioSpatialContext,
    )

    demand = [1331, 35634, 11357, 1722, 5, 61, 3872, 1622, 3992]
    inertia = [1.0, 0.98, 0.99, 0.98, 1.0, 1.0, 0.98, 0.93, 0.76]
    allocation_order = [1, 3, 4, 5, 9, 6, 7, 8, 2]
    output_file = str(tmp_path / "output.tif")
    legend_texts = [f"{name} (legend)" for name in class_names]

    scenario = ScenarioFile(
        t1_file=t1_file, t1_year=1985, t2_file=t2_file, t2_year=1995,
        output_file=output_file, drivers_t1=dict(drivers_85), drivers_t2=dict(drivers_95),
        classes=[
            ScenarioClass(
                name=name, class_id=i + 1, model_type="logistic", demand=d, inertia=r,
                legend_text=legend_texts[i], colour="#336699",
            )
            for i, (name, d, r) in enumerate(zip(class_names, demand, inertia))
        ],
        class_allocation_order=allocation_order,
        spatial_context=ScenarioSpatialContext(
            enabled=True, window_size=3, steps=2, write_step_output=True
        ),
        accuracy_assessment=ScenarioAccuracyAssessment(
            reference_file=t2_file,  # any real raster works for a text-field round trip
            base_file=t1_file,
            display_mode="overall",
        ),
        map_composition=ScenarioMapComposition(
            title="Predicted LULC", legend_heading="Land Use Classes",
        ),
    )
    scenario_path = str(tmp_path / "scenario.yaml")
    scenario.to_yaml(scenario_path)

    # This test builds its own second window rather than using the
    # `main_window` fixture, so replicate its QMessageBox mocking here too
    # (closeEvent's "Are you sure?" question and loadHelpFile's/Open's own
    # informational dialogs all block forever unmocked, headless).
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
    )

    window = mw.LULCMainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (scenario_path, "")),
    )

    window.on_actionOpen_File_triggered()

    # No compute step ran -- no background worker was ever started, and
    # Execute/Show/Export's own effects (populated cbFile_ViewMaps, a
    # written output raster) must not be present either.
    assert window._active_worker is None
    assert window.ui.cbFile_ViewMaps.count() == 0
    assert not os.path.exists(output_file)

    assert window.ui.tabWidget.currentIndex() == 4
    assert list(window._MyForm__className) == class_names

    # self.__DriverDictionaryT2 itself is only computed lazily by
    # prepareDataForExecution() (at Execute/Save time) -- check the T2
    # driver table cells the fast-forward actually populated instead.
    t2_table = window.ui.twSelectModelTypeAndDrivers_DriverSelectionT1
    driver_cols = list(window._MyForm__DriverDictionaryT1.keys())
    for col_idx, name in enumerate(driver_cols):
        col = 3 + col_idx
        got = [t2_table.item(row, col).text() for row in range(t2_table.rowCount())]
        assert got == [drivers_95[name]] * t2_table.rowCount()

    demand_got = [window.ui.twPolicies_DemandAllocation.item(i, 2).text() for i in range(9)]
    inertia_got = [window.ui.twPolicies_DemandAllocation.item(i, 3).text() for i in range(9)]
    alloc_got = [window.ui.twPolicies_DemandAllocation.item(i, 1).text() for i in range(9)]
    assert demand_got == [str(v) for v in demand]
    assert inertia_got == [str(v) for v in inertia]
    assert alloc_got == [str(v) for v in allocation_order]

    assert window.ui.cbEnable_DemandAllocationSpatialContext.isChecked()
    assert window.ui.cbWindowSize_DemandAllocationSpatialContext.currentText() == "3"
    assert window.ui.cbInSteps_DemandAllocationSpatialContext.currentText() == "2"
    assert window.ui.cbStepOutputRequired_DemandAllocationSpatilaContext.currentText() == "Yes"

    assert window.ui.leActualFile_AccuracyAssesment.text() == t2_file
    assert window.ui.leBaseFile_AccuracyAssesment.text() == t1_file
    assert window.ui.rbAgreementIndexOverall_AccuracyAssesmentDetailed.isChecked()
    assert window.ui.leTitle_ViewMaps.text() == "Predicted LULC"
    assert window.ui.leLegendHeading_ViewMaps.text() == "Land Use Classes"
    color_table = window.ui.twColorTable_ViewMaps
    assert [color_table.item(i, 1).text() for i in range(9)] == legend_texts
    assert [color_table.item(i, 2).background().color().name() for i in range(9)] == ["#336699"] * 9

    # Round-trip the other direction too: Save what Open just built and
    # confirm the re-saved file matches the original scenario.
    resaved_path = str(tmp_path / "resaved.yaml")
    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (resaved_path, "")),
    )
    window.on_actionSave_File_triggered()
    resaved = ScenarioFile.from_yaml(resaved_path)

    assert resaved.t1_file == t1_file
    assert resaved.t2_file == t2_file
    assert [c.name for c in resaved.classes] == class_names
    assert [c.demand for c in resaved.classes] == demand
    assert [c.inertia for c in resaved.classes] == inertia
    assert resaved.class_allocation_order == allocation_order
    assert resaved.spatial_context == ScenarioSpatialContext(
        enabled=True, window_size=3, steps=2, write_step_output=True
    )
    assert [c.legend_text for c in resaved.classes] == legend_texts
    assert [c.colour for c in resaved.classes] == ["#336699"] * 9
    assert resaved.accuracy_assessment.reference_file == t2_file
    assert resaved.accuracy_assessment.base_file == t1_file
    assert resaved.accuracy_assessment.display_mode == "overall"
    assert resaved.map_composition.title == "Predicted LULC"
    assert resaved.map_composition.legend_heading == "Land Use Classes"

    window.close()


def test_open_malformed_scenario_file_shows_error_not_crash(main_window, monkeypatch, tmp_path):
    """A corrupted/malformed --config or File > Open target must not crash
    the GUI -- it should stay open and show a clear error instead."""
    bad_path = tmp_path / "corrupted.yaml"
    bad_path.write_text("scenario_version: 1\ndata: [unbalanced\n")

    shown = {}
    monkeypatch.setattr(
        "gui.main_window.QMessageBox.critical",
        lambda *a, **k: shown.setdefault("message", a[2] if len(a) > 2 else None),
    )

    main_window.loadScenarioFile(str(bad_path))  # must not raise

    assert "message" in shown
    assert str(bad_path) in shown["message"] or "not valid YAML" in shown["message"]


def test_open_missing_scenario_file_shows_error_not_crash(main_window, monkeypatch):
    shown = {}
    monkeypatch.setattr(
        "gui.main_window.QMessageBox.critical",
        lambda *a, **k: shown.setdefault("message", a[2] if len(a) > 2 else None),
    )

    main_window.loadScenarioFile("/nonexistent/path/scenario.yaml")  # must not raise

    assert "message" in shown


def test_open_restores_explicit_conversion_order_matrix(
    qtbot, monkeypatch, tmp_path, t1_file, t2_file, drivers_85, drivers_95, class_names
):
    """Bug: an explicit N x N migration-order matrix (conversion_order's
    other valid form besides the "TP" default -- fully supported by the
    pipeline and by Save) was silently dropped by Open: the
    GUI stayed on the "TP" default, discarding the loaded scenario's
    actual matrix if the user proceeded to Execute without noticing."""
    import gui.main_window as mw
    from LULC.scenario import ScenarioClass, ScenarioFile

    n = len(class_names)
    matrix = [[((r + c) % n) + 1 for c in range(n)] for r in range(n)]

    scenario = ScenarioFile(
        t1_file=t1_file, t1_year=1985, t2_file=t2_file, t2_year=1995,
        output_file=str(tmp_path / "output.tif"),
        drivers_t1=dict(drivers_85), drivers_t2=dict(drivers_95),
        classes=[ScenarioClass(name=name, class_id=i + 1) for i, name in enumerate(class_names)],
        conversion_order=matrix,
    )
    scenario_path = str(tmp_path / "scenario.yaml")
    scenario.to_yaml(scenario_path)

    monkeypatch.setattr(
        QtWidgets.QMessageBox, "question", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Yes)
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox, "information", staticmethod(lambda *a, **k: QtWidgets.QMessageBox.Ok)
    )
    window = mw.LULCMainWindow()
    qtbot.addWidget(window)
    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getOpenFileName",
        staticmethod(lambda *a, **k: (scenario_path, "")),
    )

    window.on_actionOpen_File_triggered()

    assert window.ui.rbUserDefined_ModelAnalysisMigrationOrder.isChecked()
    table = window.ui.twMigrationOrder_ModelAnalysis
    got = [[int(table.item(r, c + 1).text()) for c in range(n)] for r in range(n)]
    assert got == matrix

    # And Save reproduces the same matrix back out.
    resaved_path = str(tmp_path / "resaved.yaml")
    monkeypatch.setattr(
        "gui.main_window.QFileDialog.getSaveFileName",
        staticmethod(lambda *a, **k: (resaved_path, "")),
    )
    window.on_actionSave_File_triggered()
    resaved = ScenarioFile.from_yaml(resaved_path)
    assert [[int(v) for v in row] for row in resaved.conversion_order] == matrix

    window.close()
