#!/usr/bin/env python3
"""
runLULCgui5_final.py - COMPLETE PRODUCTION VERSION
✅ Fixed pyqtSlot import
✅ HTML Help (QTextBrowser - crash-proof)
✅ All your signals + file dialogs
✅ Lazy imports (no circular)
✅ GPU disabled (macOS safe)
✅ Full LULC pipeline integration
"""


import sys
import os
import traceback
from pathlib import Path


# CRITICAL: Disable GPU/WebEngine BEFORE imports
os.environ['QTWEBENGINE_CHROMIUM_FLAGS'] = '--disable-gpu --no-sandbox --disable-gpu-sandbox'
os.environ['QT_QUICK_BACKEND'] = 'software'


from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (QApplication, QMainWindow, QFileDialog, QMessageBox,
                             QProgressDialog, QTextBrowser)


import UI.resources_rc


# Project paths
ROOT = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(ROOT))


# Safe imports
from LULC.config import logger, NA_VALUE
LULCAlgorithms = None
from UI.LULCgui import Ui_LULCModel  # Your exact UI


class LULCApplication(QMainWindow, Ui_LULCModel):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.logger = logger
        self.project_dir = ""

        # Replace crashing Help tab
        self._setup_html_help()

        # Connect YOUR signals
        self._connect_signals()

        # Lazy backend load
        global LULCAlgorithms
        import LULC.LULCAlgorithms
        LULCAlgorithms = LULCAlgorithms


    def _setup_html_help(self):
        """Crash-proof HTML Help with QTextBrowser: load OpenLDM.html from disk"""
        html_path = Path(__file__).parent / "UI/docs/OpenLDM.html"  # src/UI/docs/OpenLDM.html
        if html_path.exists():
            try:
                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()
            except Exception as e:
                html_content = f"<h1>Error loading help</h1><p>{e}</p>"
        else:
            html_content = """
            <h1>OpenLDM Help Missing</h1>
            <p>Please place a file:</p>
            <pre>{html_path}</pre>
            <p>with your HTML documentation.</p>
            """  # Fallback message

        # QTextBrowser = perfect WebEngine replacement
        browser = QTextBrowser(self.tabHelp)
        browser.setHtml(html_content)
        browser.setOpenExternalLinks(True)
        browser.setOpenLinks(False)  # Stay in app

        # Rich styling
        browser.setStyleSheet("""
            QTextBrowser {
                font-size: 11.5pt;
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #f8fafc, stop:1 white);
                border: 1px solid #e2e8f0;
                border-radius: 8px;
                padding: 15px;
                selection-background-color: #3b82f6;
            }
            QScrollBar:vertical { width: 12px; }
        """)

        # Replace in Help tab layout
        layout = self.tabHelp.layout()
        if layout:
            while layout.count():
                item = layout.takeAt(0)
                if item.widget():
                    item.widget().setParent(None)
            layout.addWidget(browser)
        else:
            if hasattr(self, 'webViewHelp'):
                self.webViewHelp.setParent(None)
            self.tabHelp.setLayout(QtWidgets.QVBoxLayout())
            self.tabHelp.layout().addWidget(browser)


    def _connect_signals(self):
        """Your exact signals"""
        signals = [
            ('pbExecute_DemandAllocation', self.on_pbExecute_DemandAllocation_clicked),
            ('pbViewModelStatistics_DriverSelectionT0DoModelFitting', self.on_pbViewModelStatistics_DriverSelectionT0DoModelFitting_clicked),
            ('pbFindAccuracy_AccuracyAssesment', self.on_pbFindAccuracy_clicked),
            ('pbSelectDirectory_DataPreparationInputSectionProjectSection', self.on_select_project_dir),
            ('pbSelectFileT0File_DataPreparationInputSelectionDataInput', self.on_select_t0),
            ('pbSelectFileT1File_DataPreparationInputSelectionDataInput', self.on_select_t1),
        ]
        for attr, func in signals:
            try:
                widget = getattr(self, attr)
                widget.clicked.connect(func)
            except AttributeError:
                continue  # Disconnectable signal missing in this UI
            except Exception as e:
                self.logger.error(f"Failed to connect {attr}: {e}")


    @pyqtSlot()
    def on_pbExecute_DemandAllocation_clicked(self):
        """Full LULC pipeline"""
        progress = QProgressDialog("Executing...", "Cancel", 0, 100, self)
        progress.setMinimumDuration(500)
        progress.show()

        try:
            # Validate inputs
            if not all([self.project_dir,
                       self.leT0File_DataPreparationInputSectionDataInput.text(),
                       self.leT1File_DataPreparationInputSectionDataInput.text()]):
                raise ValueError("Fill Project Dir + T0/T1 files!")

            progress.setValue(10)
            modelType = self._get_model_type()
            T1File = self.leT1File_DataPreparationInputSectionDataInput.text()
            T2File = self.leT0File_DataPreparationInputSectionDataInput.text()
            na_value = float(self.leNAValue_DataPreparationInputSectionProjectSection.text() or NA_VALUE)
            outputfile = self.leOutputFile_DataPreparationOutputSection.text() or f"{self.project_dir}/predicted.tif"

            progress.setValue(30)

            # YOUR EXACT BACKEND CALL
            result = LULCAlgorithms.generate_predicted_map(
                modelType=modelType,
                T1File=T1File,
                T2File=T2File,
                withClassName=True,
                T1drivers=self.project_dir,
                T2drivers=self.project_dir,
                na_value=na_value,
                demand=None,
                restrictSpatialMigration=self.leMaskDataPreparationInputSectionDataInput.text() or "",
                neighbour=None,
                outputfile=outputfile,
                conversionOrder="TP",
                classAllocationOrder=[],
                maskFile=self.leMaskDataPreparationInputSectionDataInput.text() or "",
                aoiFile=self.leAreaOfInterestDataPreparationInputSectionDataInput.text() or "",
                modelformula=None,
                suitabilityFileDirectory=self.project_dir,
            )

            progress.setValue(100)
            self.textBrowser2.append(f"✅ SUCCESS!\nOutput: {outputfile}")
            QMessageBox.information(self, "Complete!", f"LULC map generated:\n{outputfile}")

        except Exception as e:
            tb = traceback.format_exc()
            self.logger.error(tb)
            self.textBrowser2.append(f"❌ Pipeline error:\n{e}")
            QMessageBox.critical(self, "Error", str(e))
        finally:
            progress.close()


    @pyqtSlot()
    def on_pbViewModelStatistics_DriverSelectionT0DoModelFitting_clicked(self):
        try:
            summary = LULCAlgorithms.get_model_fit_summary(
                self.leT1File_DataPreparationInputSectionDataInput.text(),
                self.leT0File_DataPreparationInputSectionDataInput.text(),
                self.project_dir,
                self._get_model_type()
            )
            self.teModelparameter_OutputDriverSelectionT0DoModelFitting.setPlainText(str(summary))
        except Exception as e:
            QMessageBox.critical(self, "Model Error", str(e))


    @pyqtSlot()
    def on_pbFindAccuracy_clicked(self):
        try:
            result = LULCAlgorithms.get_kappa_summary(
                self.leActual_FileAccuracyAssesment.text(),
                self.lePredicted_FileAccuracyAssesment.text(),
                NA_VALUE,
                []
            )
            kappa = result.get('k_standard_overall', 0)
            self.leOverall_KappaAccuracyAssesment.setText(f"{kappa:.4f}")
        except Exception as e:
            QMessageBox.critical(self, "Accuracy Error", str(e))


    def _get_model_type(self):
        try:
            if hasattr(self, 'rbRandomForestDriverSelectionT0DoModelFitting') and \
               self.rbRandomForest_DriverSelectionT0DoModelFitting.isChecked():
                return "randomForest"
            return "logistic"
        except Exception:
            return "logistic"


    @pyqtSlot()
    def on_select_project_dir(self):
        path = QFileDialog.getExistingDirectory(self, "Select Project Directory")
        if path:
            self.leProjectDirectory_DataPreparationInputSectionProjectSection.setText(path)
            self.project_dir = path


    @pyqtSlot()
    def on_select_t0(self):
        path, _ = QFileDialog.getOpenFileName(self, "T0 Map", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.leT0File_DataPreparationInputSectionDataInput.setText(path)


    @pyqtSlot()
    def on_select_t1(self):
        path, _ = QFileDialog.getOpenFileName(self, "T1 Map", "", "GeoTIFF (*.tif *.tiff)")
        if path:
            self.leT1File_DataPreparationInputSectionDataInput.setText(path)


def main():
    app = QApplication(sys.argv)

    # Make sure AA_ShareOpenGLContexts is set before the QApplication runs
    app.setAttribute(QtCore.Qt.AA_ShareOpenGLContexts, False)

    window = LULCApplication()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
