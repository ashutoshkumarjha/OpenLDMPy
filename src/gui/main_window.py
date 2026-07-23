#python 3.10+
import sys, os, random, time, re #,imp
import shutil,tempfile
import os.path
from os.path import basename, splitext
from array import array
from datetime import datetime
from collections import OrderedDict

#################QT GUI IMPORTS#################
from PyQt5 import QtCore, QtGui,QtNetwork,QtWidgets
from PyQt5.QtCore import Qt,QFileInfo,QFile,QUrl
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtGui import QColor,QBrush
from PyQt5.QtWidgets import QLineEdit,QApplication,QTableWidgetItem,QPushButton,QLabel,QFileDialog,QComboBox,QMessageBox,QColorDialog
#from PyQt5.QtWebEngineWidgets import QWebEngineView
from .OpenLDMgui import Ui_LULCModel
from math import sqrt, isnan

################Resource File###################
from . import OpenLDMgui_rc

#################PURE-PYTHON BACKEND (replaces rpy2)####################
from .controller import PipelineController
from .workers import BackgroundTaskWorker
from . import log_bridge
from .progress_bridge import QtProgressRelay
from LULC.errors import PipelineError
from LULC.scenario import (
    ScenarioAccuracyAssessment,
    ScenarioClass,
    ScenarioFile,
    ScenarioFileError,
    ScenarioMapComposition,
    ScenarioSpatialContext,
)

#################FOR MAP Display and Print########



__consti = 0


class StatusBarThread(QtCore.QThread):
    trigger = QtCore.pyqtSignal(int)

    def __init__(self,statusBar, parent=None):
        QtCore.QThread.__init__(self)
        self.statusBar=statusBar
        self.startTime = datetime.now();
        self.message=""
        self._stop = False


    def setup(self, thread_no):
        self.thread_no = thread_no

    def stop(self):
        """Real stop for this thread's raw while-loop (run() never calls
        self.exec_(), so it has no Qt event loop of its own — QThread.exit()
        is a no-op here, both in this port and in the original R GUI it was
        carried over from). Called from on_actionExit_triggered instead."""
        self._stop = True

    def run(self):
        while not self._stop:
            time.sleep(1)
            timediff=datetime.now()-self.startTime
            self.message="TotalElapsed Time(sec)"+str(timediff.seconds)

    def getMessage(self):
        return(self.message)


class MyForm(QtWidgets.QMainWindow):

    ###############################333
    #Add all the save variable here
    global __consti
    __projectDirectory = "."
    __currentDirectory="."
    __T0File = ""
    __T1File = ""
    __T0Year = 0
    __T1Year = 0
    __OutputFile= "2005.tif"
    __shpfileT0 = ""
    __shpfileT1 = ""
    __checkOnScreen = 1
    __neughbourl=[]
    __DriverDictionaryT1 = {}
    __DriverDictionaryT2 = {}
    __modelNAValue=None
    __driversT1 = []
    __transitionMatrix=[]
    __pontiusResult=None
    __noOfClasses=0
    __className=[]
    customlist = []
    res = ""
    Coeffcient = []
    __T1File = ""
    __T0File = ""
    __MASKFile=""
    __AOIFile=""
    __modelformula=""
    __modeltype=""
    __Drivername = []
    noOfDrivers = 0
    __confidenceinterval=[]
    __demand=[]
    processingstep="Data Preparation"
    plot="onscreen"
    __installedDir="."
    __debug=1
    __suitabilityFileDirectory=""
    __ReferenceFile=None
    currentLogTime="19760101000000"

    def __init__(self, parent=None, embedded=False):
        QtWidgets.QWidget.__init__(self, parent)
        # True when launched inside a host application (e.g. the QGIS
        # plugin) that owns the QApplication itself -- on_actionExit_triggered
        # must not call QApplication.quit() in that case, or File > Exit
        # inside this window would kill the host application too.
        self.__embedded = embedded
        self.ui = Ui_LULCModel()
        self.ui.setupUi(self)
        #All the new objects here as self.ui.
        self.modelSummary={}
        self.controller = PipelineController()
        self._active_worker = None  # keep a reference so QThread isn't GC'd mid-run
        self.currentLogTime = datetime.now().strftime('%Y%m%d%H%M%S')
        self.prepareExecutionEnv()
        self.initModelParam()
        self.initGui()
        self.initStatusBar()
        self.initLogBridge()

    def initLogBridge(self):
        """Real, live progress text from the LULC package's own logger —
        replaces the R backend's non-functional setRStatus/
        getRStatus and the original GUI's fake elapsed-time-only
        status bar."""
        self._log_handler = log_bridge.attach()
        self._log_handler.message_logged.connect(self._on_pipeline_log)

    def _on_pipeline_log(self, message, levelno):
        self.statusBar().showMessage(message)
        if hasattr(self.ui, "teLog"):
            self.ui.teLog.appendPlainText(message)

    def _on_pipeline_progress(self, percent, label):
        """Real, quantitative progress (LULCAlgorithms.run_pipeline's own
        stage boundaries) — complements _on_pipeline_log's live text with an
        actual percentage on the progress bar."""
        self.ui.progressBar.setProperty("value", percent)
        self.statusBar().showMessage(f"{label} ({percent}%)")

    def _run_in_background(self, func, on_success, busy_widgets=(), **kwargs):
        """Run ``func(**kwargs)`` on a :class:`BackgroundTaskWorker`, disabling
        ``busy_widgets`` for the duration and re-enabling them afterwards
        either way. ``on_success(result)`` runs on the GUI thread on success;
        failures show a QMessageBox instead of crashing."""
        for w in busy_widgets:
            w.setEnabled(False)

        worker = BackgroundTaskWorker(func, **kwargs)

        def _finish(result):
            for w in busy_widgets:
                w.setEnabled(True)
            on_success(result)

        def _fail(exc):
            for w in busy_widgets:
                w.setEnabled(True)
            message = str(exc) if isinstance(exc, PipelineError) else f"{type(exc).__name__}: {exc}"
            QMessageBox.critical(self, "OpenLDM Error", message)

        worker.finished_ok.connect(_finish)
        worker.failed.connect(_fail)
        self._active_worker = worker  # keep alive until it finishes
        worker.start()

    def getStatus(self):
        self.statusBar().showMessage(self.statusMessage.getMessage())

    def initStatusBar(self):
        self.statusMessage=StatusBarThread(self.ui.statusBar)
        self.statusMessage.start()
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.getStatus)
        # check every second
        self.timer.start(1000*1)

    def initGui(self):
        self.setGeometry(50,50,900,600)
        self.resize(900,600)
        self.setWindowTitle("Open-source Land-use Land-cover Dynamics Modeling Platfrom ver-1.0")
        self.ui.leOutputFile_DataPreparationOutputSection.setText("")
        self.ui.twSelectDrivers_DriverSelectionT0.resizeColumnsToContents()
        self.loadHelpFile()

    def initModelParam(self):
        self.__demand=None
        self.__modelNAValue=None
        self.__MASKFile=None
        self.__AOIFile=None
        self.__suitabilityFileDirectory=None

    #############################################################################
    def loadHelpFile(self):
        fileName=self.__installedDir+"/helpdoc/OpenLDM.html"
        fd = QFile(fileName)
        if not fd.open(QtCore.QIODevice.ReadOnly):
            # NB: original code called the nonexistent QtGui.QMessageBox
            # (QMessageBox lives in QtWidgets) — fixed here.
            QtWidgets.QMessageBox.information(self, "Unable to open file",fd.errorString())
            return

        output = QtCore.QTextStream(fd).readAll()
        # Display contents.
        self.setBaseUrl(QtCore.QUrl.fromLocalFile(fileName))
        self.ui.webView_Help.setHtml(output, self.baseUrl)

    def setBaseUrl(self, url):
        self.baseUrl = url

    def setNA(self):
        navalue=self.ui.leNAValue_DataPreparationInputSectionProjectSection.text()
        if(navalue=='NA' or len(navalue)==0):
            self.__modelNAValue=None
        else:
            self.__modelNAValue=int(self.ui.leNAValue_DataPreparationInputSectionProjectSection.text())

    def update_text(self, thread_no):
        QtGui.QApplication.processEvents()
        time.sleep(random.uniform(0,0.7))
        self.ui.tbLog.setText(str(self.__check)+"% Completed")
        self.__check = self.__check + 1
        if(self.__check == 100):
            self.ui.leOutputFile.setText(str(self.__currentDirectory)+"/"+self.__T0File+".tif")
        QtGui.QApplication.processEvents()

    #Add all the requuired signles Here
    @pyqtSlot()
    def on_pbSelectDirectory_DataPreparationInputSectionProjectSection_clicked(self):
        self.__projectDirectory = QFileDialog.getExistingDirectory(self, "Open Project Directory",".", QFileDialog.ShowDirsOnly);
        self.ui.leProjectDirectory_DataPreparationInputSectionProjectSection.setText(str(self.__projectDirectory))

    @pyqtSlot()
    def on_pbSelectFileT0File_DataPreparationInputSelectionDataInput_clicked(self):
        file,_ = QFileDialog.getOpenFileName(self, "Open File",
                                                           self.__projectDirectory,"Raster (*.img *.tif );;Esri Shape (*.shp)");
        (dirName, fileName) = os.path.split(str(file))
        self.__currentDirectory=dirName
        self.__T0File=splitext(fileName)[0]
        fileType = splitext(fileName)[1]
        if(fileType == ".tif" or fileType == ".img"):
            self.__T0File=str(file)
            self.enable_ValidateDataPreparation()
        elif(fileType == ".shp" ):
            self.__shpfileT0 = str(file)
            self.__T0File=self.__T0File+".tif"
            self.ui.pbConvert_T0_DataPreparationInputSectionDataInput.setEnabled(True)
            self.ui.sbGridsize_DataPreparationInputSectionProjectSection.setEnabled(True)
        self.ui.leT0File_DataPreparationInputSectionDataInput.setText(str(self.__T0File))


    @pyqtSlot()
    def on_pbSelectFileT1File_DataPreparationInputSelectionDataInput_clicked(self):
        file,_ = QtWidgets.QFileDialog.getOpenFileName(self, "Open File",
                                                           self.__currentDirectory,"Raster (*.img *.tif );;Esri Shape (*.shp)");
        #filename,_=file.filename()
        print(file)

        (dirName, fileName) = os.path.split(str(file))
        self.__T1File=splitext(fileName)[0]
        fileType = splitext(fileName)[1]
        if(fileType == ".tif" or fileType == ".img"):
            self.__T1File=str(file)
            self.ui.leT1File_DataPreparationInputSectionDataInput.setText(str(self.__T1File))
            self.enable_ValidateDataPreparation()
        else:
            self.__shpfileT1 = str(file)
            self.__T1File=self.__T1File+".tif"
            self.ui.pbConvert_T1_DataPreparationInputSectionDataInput.setEnabled(True)
            self.ui.sbGridsize_DataPreparationInputSectionProjectSection.setEnabled(True)
        self.ui.leT1File_DataPreparationInputSectionDataInput.setText(str(self.__T1File))


    @pyqtSlot()
    def on_pbSelectFileOutputFile_DataPreparationOutputSection_clicked(self):
        fileLocation = QFileDialog.getSaveFileName(self, "Output File",
                                                                 self.__currentDirectory,"Raster (*.tif )")[0];
        (dirName, fileName) = os.path.split(str(fileLocation))
        print(dirName, fileName)
        fileType = splitext(fileName)[1]
        if(fileType == ".tif"):# or fileType == ".img"):
            self.__OutputFile=str(fileLocation)
        else:
            self.__OutputFile = str(fileLocation)+".tif"
        self.ui.leOutputFile_DataPreparationOutputSection.setText(self.__OutputFile)
        self.ui.lePredictedFile_AccuracyAssesment.setText(self.__OutputFile)
        self.enable_ValidateDataPreparation()



    def _convert_shapefile(self, shp_file, is_t0):
        """Rasterize a T0/T1 shapefile via LULC.rasterize.
        Runs in the background since fine grid sizes can be slow."""
        if not shp_file:
            QMessageBox.information(
                self, "No shapefile selected", "Select a .shp file for T0/T1 first."
            )
            return
        grid_size = self.ui.sbGridsize_DataPreparationInputSectionProjectSection.value()
        output_file = splitext(shp_file)[0] + ".tif"

        def _on_success(path):
            if is_t0:
                self.__T0File = path
                self.ui.leT0File_DataPreparationInputSectionDataInput.setText(path)
            else:
                self.__T1File = path
                self.ui.leT1File_DataPreparationInputSectionDataInput.setText(path)
            self.enable_ValidateDataPreparation()
            QMessageBox.information(self, "Conversion complete", f"Wrote {path}")

        busy_widgets = (
            self.ui.pbConvert_T0_DataPreparationInputSectionDataInput,
            self.ui.pbConvert_T1_DataPreparationInputSectionDataInput,
            self.ui.sbGridsize_DataPreparationInputSectionProjectSection,
        )
        self._run_in_background(
            self.controller.rasterize_shapefile, _on_success, busy_widgets=busy_widgets,
            shp_file=shp_file, output_file=output_file, grid_size=grid_size,
        )

    @pyqtSlot()
    def on_pbConvert_T0_clicked(self):
        self._convert_shapefile(self.__shpfileT0, is_t0=True)

    @pyqtSlot()
    def on_pbConvert_T1_clicked(self):
        self._convert_shapefile(self.__shpfileT1, is_t0=False)

    @pyqtSlot()
    def on_pbSelectFileAreaOfInterest_DataPreparationInputSectionDataInput_clicked(self):
        file,_ = QFileDialog.getOpenFileName(self, "Open File",
                                                           self.__currentDirectory,"ESRI (*.shp )");
        (dirName, fileName) = os.path.split(str(file))
        self.__currentDirectory=dirName
        self.ui.leAreaOfInterest_DataPreparationInputSectionDataInput.setText(str(file))

    @pyqtSlot()
    def on_pbSelectFileMask_DataPreparationInputSectionDataInput_clicked(self):
        file,_ = QFileDialog.getOpenFileName(self, "Open File",
                                                       self.__currentDirectory,"ESRI (*.shp )");
        (dirName, fileName) = os.path.split(str(file))
        self.__currentDirectory=dirName
        self.ui.leMask_DataPreparationInputSectionDataInput.setText(str(file))

    def getCurrentDirectory():
        return(self.__currentDirectory)


    @pyqtSlot()
    def on_pbNextDataPreparation_clicked(self):
        self.setNA()
        self.ui.tabWidget.setCurrentIndex(1)
        self.ui.progressBar.setProperty("value",10)
        self.preparecbInSteps_DemandAllocationSpatialContext()
        self.ui.gbSelectDrivers_DriverSelectionT0.setEnabled(True)
        self.ui.pbAddDriver_DriverSelectionT0SelectDrivers.setEnabled(True)
        self.ui.twSelectDrivers_DriverSelectionT0.setEnabled(True)


    #################Module:2####################



    @pyqtSlot() # prevents executing following function twice
    def SelectDriver_pushed(self):
        sending_button = str(self.sender().objectName())
        rowstr = sending_button[sending_button.index('SelectDriver')+12:]
#Object name is suffiexed by the driver number and row no is less than one of that
        row = int(rowstr)-1
        filename,_ = QFileDialog.getOpenFileName(self, "Open File",self.__currentDirectory,"Raster (*.tif *.img)");
        (dirName, OnlyFilename) = os.path.split(filename.strip())
        self.__currentDirectory=dirName
        disp=self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(row, 1)
        disp.setText(filename)
        disp=self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(row, 0)
        disp.setText(OnlyFilename.replace(".","_"))
        self.ui.twSelectDrivers_DriverSelectionT0.resizeColumnsToContents()


    @QtCore.pyqtSlot() # prevents executing following function twice
    def DeleteDriver_pushed(self):
        sending_button = str(self.sender().objectName())
        rowstr = sending_button[sending_button.index('DeleteDriver')+12:] #Object name is suffiexed by the driver number and row no is less than one of that
        row = int(rowstr)
        lastRow=self.ui.twSelectDrivers_DriverSelectionT0.rowCount();
        if(row==lastRow):
            reply= QtWidgets.QMessageBox.question(self, "Delete Row","Are You Sure?",QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No);
            if(reply==QtWidgets.QMessageBox.Yes ):
                self.ui.twSelectDrivers_DriverSelectionT0.resizeColumnsToContents()
                disp=self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(row, 1)
                self.ui.twSelectDrivers_DriverSelectionT0.removeRow(row-1)
        else:
            QtWidgets.QMessageBox.about(self, "Delete Last Row","Only Last Row can be deleted");
        self.ui.twSelectDrivers_DriverSelectionT0.resizeColumnsToContents()
        if (lastRow<4):
            self.ui.gbDoModelFitting_DriverSelectionT0.setEnabled(False)
        else:
            self.ui.gbDoModelFitting_DriverSelectionT0.setEnabled(True)

    @pyqtSlot()
    def on_pbAddDriver_DriverSelectionT0SelectDrivers_clicked(self):
        #Append New Row
        workingRow=self.ui.twSelectDrivers_DriverSelectionT0.rowCount()+1
        workingRowIdx=workingRow-1
        self.ui.twSelectDrivers_DriverSelectionT0.setRowCount(workingRow)

        #Put label for driver name
        nameLabel = QtWidgets.QLineEdit("Driver"+str(workingRow))
        self.ui.twSelectDrivers_DriverSelectionT0.setCellWidget(workingRowIdx, 0, nameLabel)

        #put Place for file name
        disp = QLabel("")
        self.ui.twSelectDrivers_DriverSelectionT0.setCellWidget(workingRowIdx, 1,disp)

        #Put Select Driver Button
        pbSelectDriver = QtWidgets.QPushButton()
        pbSelectDriver.setText("Select Driver")
        pbSelectDriver.setObjectName("SelectDriver"+str(workingRow))
        pbSelectDriver.clicked.connect(self.SelectDriver_pushed)
        self.ui.twSelectDrivers_DriverSelectionT0.setCellWidget(workingRowIdx, 2, pbSelectDriver)

        #Put Delete Driver Button
        pbDeleteDriver = QtWidgets.QPushButton()
        pbDeleteDriver.setText("Delete Driver")
        pbDeleteDriver.setObjectName("DeleteDriver"+str(workingRow))
        pbDeleteDriver.clicked.connect(self.DeleteDriver_pushed)
        self.ui.twSelectDrivers_DriverSelectionT0.setCellWidget(workingRowIdx, 3, pbDeleteDriver)

        if(self.ui.twSelectDrivers_DriverSelectionT0.rowCount()>2):
            self.ui.gbDoModelFitting_DriverSelectionT0.setEnabled(True)

        self.ui.twSelectDrivers_DriverSelectionT0.resizeColumnsToContents()

    def createModelStatisticDetails(self):
        filename=self.ui.leT0File_DataPreparationInputSectionDataInput.text();
        if (filename != self.__T0File):
            self.__T0File=filename
        filename=self.ui.leT1File_DataPreparationInputSectionDataInput.text();
        if (filename != self.__T1File):
            self.__T1File=filename
        self.__Drivername=[]
        self.__driversT1=[]
        self.__DriverDictionaryT1= OrderedDict()#{};
        for i in list(range(0,self.ui.twSelectDrivers_DriverSelectionT0.rowCount())):
            self.__Drivername.append(str(self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(i,0).text()))
            self.__driversT1.append(str(self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(i,1).text()))
            self.__DriverDictionaryT1[str(self.__Drivername[i])] = self.__driversT1[i]
        self.noOfDrivers = len(self.__DriverDictionaryT1)
        if(self.__debug==1):
            print(self.__Drivername)
            print(self.__driversT1)
            
        projectNA=self.ui.leNAValue_DataPreparationInputSectionProjectSection.text().strip()
        if (projectNA=='NA' or len(projectNA)==0):
            self.__modelNAValue=None
        else:
            self.__modelNAValue=int(projectNA)
        filename=self.ui.leAreaOfInterest_DataPreparationInputSectionDataInput.text();
        if(len(filename)!=0):
            if (filename != self.__AOIFile):
                self.__AOIFile=str(filename)
        else:
            self.__AOIFile=None
        filename=self.ui.leMask_DataPreparationInputSectionDataInput.text();
        if(len(filename)!=0):
            if (filename != self.__MASKFile):
                self.__MASKFile=str(filename)
        else:
            self.__MASKFile=None

    @pyqtSlot()
    def on_pbViewModelStatistics_DriverSelectionT0DoModelFitting_clicked(self):
        if(self.__debug==1):
            print ("on_pbViewModelStatistics_DriverSelectionT0DoModelFitting_clicked")
        self.createModelStatisticDetails();
        self.processingstep="Doing Model Summary";
        self.printParameter()

        def on_success(model_summary):
            self.modelSummary = model_summary  # {class_name: summary_text}
            path=os.path.join(str(self.__projectDirectory),self.currentLogTime+'-summary.log')
            with open(path, "w") as fh:
                fh.write("\n\n".join(model_summary.values()))
            self.ui.teModelparameterOutput_DriverSelectionT0DoModelFitting.setPlainText(
                "\n\n".join(model_summary.values())
            )
            self.ui.pbNext_DriverSelectionT0.setEnabled(True)
            self.ui.teModelparameterOutput_DriverSelectionT0DoModelFitting.setEnabled(True)

        self._run_in_background(
            self.controller.get_model_fit_summary,
            on_success,
            # All model-type radios lock for the duration too, not just the
            # button — otherwise switching model type mid-fit re-enables
            # "View Model Statistics" (each on_rbXXX_..._toggled handler
            # does that unconditionally) and lets a second background fit
            # start while the first is still running. Next locks too: it
            # can already be enabled from an earlier successful fit, and
            # clicking through mid-fit shouldn't be possible either.
            busy_widgets=[
                self.ui.pbViewModelStatistics_DriverSelectionT0DoModelFitting,
                self.ui.pbNext_DriverSelectionT0,
                self.ui.rbLogisticRegression_DriverSelectionT0DoModelFitting,
                self.ui.rbLinearRegression_DriverSelectionT0DoModelFitting,
                self.ui.rbNeuralregression_DriverSelectionT0DoModelFitting,
                self.ui.rbRandomForest_DriverSelectionT0DoModelFitting,
                self.ui.rbSVM_DriverSelectionT0DoModelFitting,
            ],
            t1_file=self.__T0File,
            t2_file=self.__T1File,
            t1_drivers=self.__DriverDictionaryT1,
            model_type=str(self.__modeltype),
            na_value=self.__modelNAValue,
            method="NotIncludeCurrentClass",
            mask_file=self.__MASKFile,
            aoi_file=self.__AOIFile,
        )

    def buildtwSelectModelTypeAndDrivers_DriverSelectionT1(self,twSelectModelTypeAndDrivers):
        twSelectModelTypeAndDrivers.setColumnCount(self.noOfDrivers+3)
        twSelectModelTypeAndDrivers.setRowCount(self.__noOfClasses)
        row=twSelectModelTypeAndDrivers.rowCount()
        col=twSelectModelTypeAndDrivers.columnCount()
        twSelectModelTypeAndDrivers.blockSignals(True)
        for i in list(range(0, row, 1)):
            for j in list(range(0, col, 1)):
                item = QTableWidgetItem()
                if(j!=0):
                    item.setFlags(item.flags()^QtCore.Qt.ItemIsEditable)
                else:
                    item.setText(self.__className[i])
                    label=QLabel(self.getSelectedModel())
                    self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setCellWidget(i, 2,label)
                twSelectModelTypeAndDrivers.setItem(i, j, item)
        twSelectModelTypeAndDrivers.blockSignals(False)

        for i in list(range(0, row, 1)):
            for j in list(range(3, col, 1)):
                item = twSelectModelTypeAndDrivers.item(i,j)
                item.setFlags(QtCore.Qt.ItemIsDragEnabled|QtCore.Qt.ItemIsEnabled)
                item.setCheckState(QtCore.Qt.Checked)

        stringlist1 = list()#QtCore.QStringList()
        stringlist1.append('Class')
        stringlist1.append('DN Value')
        stringlist1.append('Model Type')
        for i in list(range(0,self.noOfDrivers,1)):
            stringlist1.append(list(self.__DriverDictionaryT1.keys())[i])

        twSelectModelTypeAndDrivers.setHorizontalHeaderLabels(stringlist1)
        twSelectModelTypeAndDrivers.resizeColumnsToContents()

    def getSelectedModel(self):
        if(self.ui.rbLogisiticRegression_ModelAnalysisViewModelCoeeffcient.isChecked()):
            modelname="Logistic Regression"
        elif(self.ui.rbLinearRegression_ModelAnalysisViewModelCoeeffcient.isChecked() ):
            modelname="Linear Regression"
        elif(self.ui.rbNeuralRegression_ModelAnalysisViewModelCoeeffcient.isChecked()):
            modelname="Neural Regression"
        elif(self.ui.rbRandomForest_ModelAnalysisViewModelCoeeffcient.isChecked()):
            modelname="Random Forest"
        elif(self.ui.rbSVM_ModelAnalysisViewModelCoeeffcient.isChecked()):
            modelname="SVM"

        return(modelname)


    def buildtwViewModelCoefficint_ModelAnalysis(self,twViewModelCoefficint):
        twViewModelCoefficint.setRowCount(self.__noOfClasses)
        twViewModelCoefficint.setColumnCount(self.noOfDrivers+3)
        #Setting Individual tabelitems
        row=twViewModelCoefficint.rowCount()
        col=twViewModelCoefficint.columnCount()
        twViewModelCoefficint.blockSignals(True)
        for i in list(range(0, row, 1)):
            for j in list(range(0, col, 1)):
                item = QTableWidgetItem()
                if(j!=0):
                    item.setFlags(item.flags()^QtCore.Qt.ItemIsEditable)
                else:
                    item.setText(self.__className[i])
                twViewModelCoefficint.setItem(i, j, item)
        twViewModelCoefficint.blockSignals(False)
        #Setting Table Header
        stringlist2 = list()#QtCore.QStringList()
        stringlist2.append('Class')
        stringlist2.append('DN Value')
        stringlist2.append('Intercept')
        for i in list(range(0,len(self.__DriverDictionaryT1),1)):
            stringlist2.append(list(self.__DriverDictionaryT1.keys())[i])

        twViewModelCoefficint.setHorizontalHeaderLabels(stringlist2)
        twViewModelCoefficint.resizeColumnsToContents()

        for i in list(range(0, row, 1)):
            for j in list(range(0, col, 1)):
                item = twViewModelCoefficint.item(i,j)
                if(j>2):
                    item.setFlags(QtCore.Qt.ItemIsDragEnabled|QtCore.Qt.ItemIsUserCheckable|QtCore.Qt.ItemIsEnabled)
                    item.setCheckState(QtCore.Qt.Checked)

    def populateDataIntoTableViewModelCoefficint_ModelAnalysis(self):
        #Populate the DriversFile in ModelAnalysis tab
        for j in list(range(0,self.ui.twViewModelCoefficint_ModelAnalysis.rowCount(),1)):
            item1 = self.ui.twViewModelCoefficint_ModelAnalysis.item(j,1)
            item1.setText(self.__className[j])
            item1.setFlags(QtCore.Qt.ItemIsDragEnabled|QtCore.Qt.ItemIsEnabled)
        
        for j in list(range(0,self.ui.twViewModelCoefficint_ModelAnalysis.rowCount(),1)):
            for k in list(range(2,self.ui.twViewModelCoefficint_ModelAnalysis.columnCount(),1)): #First three colums are reserved for classname,number and modetype
                item1 = self.ui.twViewModelCoefficint_ModelAnalysis.item(j,k)
                item1.setText(str(self.__confidenceinterval[j][k-2][1])) #
                toolstr = str(self.__confidenceinterval[j][k-2][2]) #k-3+1 since Intercept is at k-3
                item1.setToolTip(toolstr)

    @QtCore.pyqtSlot()
    def on_pbNext_DriverSelectionT0_clicked(self):
        try:
            class_ids = self.controller.get_class_codes(self.__T0File, na_value=self.__modelNAValue)
        except PipelineError as exc:
            QMessageBox.critical(self, "OpenLDM Error", str(exc))
            return
        self.__className = [str(c) for c in class_ids]
        self.__noOfClasses=len(self.__className)
        self.noOfDrivers=len(self.__Drivername)
        self.createLULCVsDriverCoefficientMatrix();
        self.buildtwMigrationOrder_ModelAnalysis(self.ui.twMigrationOrder_ModelAnalysis)
        self.buildtwSelectModelTypeAndDrivers_DriverSelectionT1(self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1)
        self.buildtwPolicies_DemandAllocation(self.ui.twPolicies_DemandAllocation);
        self.buildtwColorTable_ViewMaps(self.ui.twColorTable_ViewMaps)
        self.buildtwViewModelCoefficint_ModelAnalysis(self.ui.twViewModelCoefficint_ModelAnalysis);
        self.populateDataIntoTableViewModelCoefficint_ModelAnalysis()
        self.populateDataIntoTableModelTypeAndDriversDriverSelectionT1()
        self.ui.tabWidget.setCurrentIndex(2)
        self.ui.progressBar.setProperty("value",30)
        self.ui.gbViewModelCoefficient_ModelAnalysis.setEnabled(True)
        self.ui.gbMigrationOrder_ModelAnalysis.setEnabled(True)
        self.ui.twMigrationOrder_ModelAnalysis.setEnabled(False)
        self.setModelAnalysisModelType()
        self.ui.pbNext_ModelAnalysis.setEnabled(True)


    @pyqtSlot()
    def on_pbNext_ModelAnalysis_clicked(self):
        self.ui.gbSelectModelTypeAndDrivers_DriverSelectionT1.setEnabled(True)
        self.ui.tabWidget.setCurrentIndex(3)
        self.ui.progressBar.setProperty("value",40)
        self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setEnabled(True)
        self.ui.pbNext_DriverSelectionT1.setEnabled(True)
        self.ui.twColorTable_ViewMaps.setEnabled(True)
        
    @pyqtSlot()
    def on_pbNext_DriverSelectionT1_clicked(self):
        self.ui.tabWidget.setCurrentIndex(4)
        self.ui.progressBar.setProperty("value",50)
        self.ui.gbPolicies.setEnabled(True)
        self.ui.gbSpatialContext.setEnabled(True)
###pratu
        ##self.ui.gbSuitabilityMapGeneration.setEnabled(True)
        self.ui.leSuitablityFile_OutputGenerationSuitabilityMapGeneration.setEnabled(False)
        self.ui.pbSelectFileSuitablityFile_OutputGenerationSuitabilityMapGeneration.setEnabled(False)
        self.ui.lbSuitablityfile.setEnabled(False)
###
        self.ui.pbExecute_DemandAllocation.setEnabled(True)

    @pyqtSlot(bool)
    def on_cbclassallocation_DemandAllocation_clicked(self,state):
        tw=self.ui.twPolicies_DemandAllocation
        if(state):
            for i in list(range(0,tw.rowCount()-1,1)):
                item=tw.item(i,1)
                label=str(i+1)
                self.ui.twPolicies_DemandAllocation.item(i,1).setText(label)
                item.setFlags(item.flags()|QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEditable)


        else:
            for i in list(range(0,tw.rowCount(),1)):
                item=tw.item(i,1)
                item.setFlags(item.flags()^QtCore.Qt.ItemIsEnabled)
                
    @pyqtSlot(bool)
    def on_cbUserDefinedClassInertia_DemandAllocation_clicked(self,state):
        tw=self.ui.twPolicies_DemandAllocation
        if(state):
            for i in list(range(0,tw.rowCount()-1,1)):
                item=tw.item(i,3)
                self.ui.twPolicies_DemandAllocation.item(i,3).setText('0')
                item.setFlags(item.flags()|QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEditable)


        else:
            for i in list(range(0,tw.rowCount(),1)):
                item=tw.item(i,3)
                item.setFlags(item.flags()^QtCore.Qt.ItemIsEnabled)


    @pyqtSlot(bool)
    def on_cbUserDefinedDemand_DemandAllocation_clicked(self,state):
        tw=self.ui.twPolicies_DemandAllocation
        if(state):
            for i in list(range(0,tw.rowCount()-1,1)):
                item=tw.item(i,2)
                item.setFlags(item.flags()|QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEditable)
        else:
            for i in list(range(0,tw.rowCount(),1)):
                item=tw.item(i,2)
                item.setFlags(item.flags()^QtCore.Qt.ItemIsEnabled)
####pratu

    @pyqtSlot(QTableWidgetItem)
    def on_twPolicies_DemandAllocation_itemChanged(self,item):
        if item.column() == 0 and item.row() < self.__noOfClasses:  # class-name sync; last row is "Total"
            self._sync_class_name(item.row(), item.text())
            return
        if(item.isSelected()):
            col=self.ui.twPolicies_DemandAllocation.currentColumn()
            row=self.ui.twPolicies_DemandAllocation.currentRow()
            sum=0
            if(col==1):
                getvalue=str(item.text())
                if(getvalue.isdigit()):
                    newvalue=int(getvalue)
                    oldvalue=newvalue
                    for j in list(range(0,self.__noOfClasses,1)):
                        flag=1
                        for k in list(range(0,self.__noOfClasses,1)):
                            checkitem=int(str(self.ui.twPolicies_DemandAllocation.item(k,1).text()))
                            if((j+1)==checkitem):
                                flag=0
                                break
                        if(flag!=0):
                            oldvalue=str(j+1)
                            break
                    if(0<newvalue<=int(str(self.__noOfClasses))):
                        for i in list(range(0,self.__noOfClasses,1)):
                            if(i!=row):
                                checkitem=int(str(self.ui.twPolicies_DemandAllocation.item(i,1).text()))
                        if(checkitem==newvalue):
                            newrow=i
                            self.ui.twPolicies_DemandAllocation.item(newrow,1).setText(oldvalue)
                    else:
                        print ("please enter value within range:1-",self.__noOfClasses)
                        self.ui.twPolicies_DemandAllocation.item(row,1).setText(oldvalue)
                else:
                    print ("WRONG INPUT")
                    self.ui.twPolicies_DemandAllocation.item(row,1).setText(oldvalue)
            elif(col==3):
                r=str(item.text())
                lenn=len(r)
                if(lenn==1):
                    if not(r[0]=='0'or r[0]=='1'):
                        self.ui.twPolicies_DemandAllocation.item(row,3).setText('0')
                else:
                    if(r[0]=='0'):
                        if(str(r[1])=='.'):
                            for i in list(range (2,len(r),1)):
                                if(r[i].isdigit()):
                                    print ('')
                                else:
                                    self.ui.twPolicies_DemandAllocation.item(row,3).setText('0')

                    elif(r[0]=='.'):
                        for i in list(range (1,len(r),1)):
                            if(r[i].isdigit()):
                                print ('')
                            else:
                                self.ui.twPolicies_DemandAllocation.item(row,3).setText('0')
                    else:
                        self.ui.twPolicies_DemandAllocation.item(row,3).setText('0')

            elif(col==2):
                getvalue=str(item.text())
                if(getvalue.isdigit()):
                    for i in list(range(0,self.__noOfClasses,1)):
                        var1=str(self.ui.twPolicies_DemandAllocation.item(i,2).text())
                        if(var1):
                            sum=sum+int(var1)
                    finalsum=str(sum)
                    self.ui.twPolicies_DemandAllocation.item(self.__noOfClasses,2).setText(finalsum)
                else:
                    print ("Only Posotive Integer Allowed")
                    self.ui.twPolicies_DemandAllocation.item(row,2).setText('')
                    for i in list(range(0,self.__noOfClasses,1)):
                        var1=str(self.ui.twPolicies_DemandAllocation.item(i,2).text())
                        if(var1):
                            sum=sum+int(var1)
                    finalsum=str(sum)
                    self.ui.twPolicies_DemandAllocation.item(self.__noOfClasses,2).setText(finalsum)
            self.ui.twPolicies_DemandAllocation.setCurrentCell(row+1,col)
#####
    def getFormulaFrom(self,row):
        noOfDrivers=self.ui.twSelectDrivers_DriverSelectionT0.rowCount()
        formula=self.ui.twViewModelCoefficint_ModelAnalysis.item(row,0).text()
        formula="T1."+formula+"~"
        for j in list(range(0,noOfDrivers,1)):
            isIncludeDriver=self.ui.twViewModelCoefficint_ModelAnalysis.item(row,j+3).checkState()
            if(isIncludeDriver == QtCore.Qt.Checked):
                if(formula.endswith("~")):
                    formula=formula+ "TD1."+ self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(j,0).text()
                else:
                    formula=formula+ "+" + "TD1." + self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(j,0).text()
        return(formula)

    def prepareFormula(self):
        noOfClass=self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount()
        formulaList=[]
        for i in list(range(0,noOfClass,1)):
            formula=self.getFormulaFrom(i)
            formulaList.append(str(formula))
        return(formulaList)

    def prepareDriversT2(self):
        noOfDrivers=self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.columnCount()-3
        driverDictionary=OrderedDict()
        for j in list(range(0,noOfDrivers,1)):
            driverDictionary[str(self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(j,0).text())]=str(self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.item(0,j+3).text())
        return(driverDictionary)

    def prepareDriversT1(self):
        noOfDrivers=self.ui.twSelectDrivers_DriverSelectionT0.rowCount()
        driverDictionary=OrderedDict()
        for j in list(range(0,noOfDrivers,1)):
            driverDictionary[str(self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(j,0).text())]=str(self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(j,1).text())
        return(driverDictionary)

    def getModelFrom(self,row):
        currentItem=self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.cellWidget(row,2)#
        if(isinstance(currentItem,QLabel)):
            currentIndexItem=currentItem.text()
        else:
            currentIndexItem=currentItem.currentText()
        if(currentIndexItem == "Logistic Regression"):
            return('logistic')
        if(currentIndexItem == "Linear Regression"):
            return('regression')
        if(currentIndexItem == "Neural Regression"):
            return('nnet')
        if(currentIndexItem == "Random Forest"):
            return('randomForest')
        if(currentIndexItem == "SVM"):
            return('svm')
        return("Wrong")

    def preparemodelType(self):
        noOfClass=self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount()
        modelType=[]
        #        modelType.append('logistic')
        #else:
        #        modelType.append(str(formulanew))
        for i in list(range(0,noOfClass,1)):
            formulanew=self.getModelFrom(i)
            modelType.append(str(formulanew))        
        return(modelType)

    def prepareModelDetail(self):
        print('prepareModelDetail')
        self.__modelformula=self.prepareFormula()
        self.__DriverDictionaryT1=self.prepareDriversT1()
        self.__DriverDictionaryT2=self.prepareDriversT2()
        self.__modeltype=self.preparemodelType()
    
    @pyqtSlot(bool)
    def on_cbEnable_DemandAllocationSpatialContext_clicked(self,state):
        print('cbEnable_DemandAllocationSpatialContext')
        self.ui.cbStepOutputRequired_DemandAllocationSpatilaContext.setEnabled(state)
        self.ui.cbWindowSize_DemandAllocationSpatialContext.setEnabled(state)
        self.ui.cbStepOutputRequired_DemandAllocationSpatilaContext.setEnabled(state)
        self.ui.cbInSteps_DemandAllocationSpatialContext.setEnabled(state)
        self.ui.lbWindowSize.setEnabled(state)
        self.ui.lbInSteps.setEnabled(state)
        self.ui.lbStepOutputRequired.setEnabled(state)


    def prepareSpatialData(self):
        if(self.ui.cbEnable_DemandAllocationSpatialContext.checkState()):
            window_size=int(str(self.ui.cbWindowSize_DemandAllocationSpatialContext.currentText()))
            steps=int(str(self.ui.cbInSteps_DemandAllocationSpatialContext.currentText()))
            write_steps=str(self.ui.cbStepOutputRequired_DemandAllocationSpatilaContext.currentText())=="Yes"
            self.__neughbourl=[window_size, steps, 1 if write_steps else 0]
        else:
            self.__neughbourl=None

    def prepareClassName(self):
        self.__className=[];
        for j in list(range(0,self.__noOfClasses,1)):
            self.__className.append(str(self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.item(j,0).text()))

    def prepareDemand(self):
        self.__demand=[]
        if(self.ui.cbUserDefinedDemand_DemandAllocation.isChecked()):
            for j in list(range(0,self.__noOfClasses,1)):
                if(str(self.ui.twPolicies_DemandAllocation.item(j,2).text())):
                    self.__demand.append(int(str(self.ui.twPolicies_DemandAllocation.item(j,2).text())))
                else:
                    self.__demand.append(0)
        else:
            self.__demand=None

    def prepareInertia(self):
        self.__restricSpatial=[]
        if(self.ui.cbUserDefinedClassInertia_DemandAllocation.isChecked()):
            for j in list(range(0,self.__noOfClasses,1)):
                if(str(self.ui.twPolicies_DemandAllocation.item(j,3).text())=='0'or str(self.ui.twPolicies_DemandAllocation.item(j,3).text())=='1'):
                    self.__restricSpatial.append(int(str(self.ui.twPolicies_DemandAllocation.item(j,3).text())))
                else:
                    self.__restricSpatial.append(float(self.ui.twPolicies_DemandAllocation.item(j,3).text()))
        else:
            self.__restricSpatial=None

    def prepareConversionOrder(self):
        if(self.ui.rbUserDefined_ModelAnalysisMigrationOrder.isChecked()):
            n=self.__noOfClasses
            self.__conversionOrder=[
                [int(str(self.ui.twMigrationOrder_ModelAnalysis.item(row,col+1).text())) for col in range(n)]
                for row in range(n)
            ]
        else:
            self.__conversionOrder='TP'


    def prepareDataForExecution(self):
        self.prepareSpatialData()
        self.prepareClassName()
        self.prepareDemand()
        self.prepareInertia()
        self.prepareConversionOrder()
        self.prepareModelDetail()
        self.prepareClassAllocationOrder()
        self.prepareSuitabilityFileDirectory()
        self.prepareReferenceFile()

        print("model.type-[[ "+str(self.__modeltype)+ " ]] ")
        print("T0File-[[ "+self.__T0File+ " ]] ")
        print("T1File-[[ "+self.__T1File+ " ]] ")
        print("with.class.name-[[ "+str(self.__className)+ " ]] ")
        print("T1drivers-[[ "+str(self.__DriverDictionaryT1)+ " ]] ")
        print("T2drivers-[[ "+str(self.__DriverDictionaryT2)+ " ]] ")
        print("withNAvalue-[[ "+str(self.__modelNAValue)+ " ]] ")
        print("demand-[[ "+str(self.__demand)+ " ]] ")
        print("restrictSpatialMigration-[[ "+str(self.__restricSpatial) + " ]] ")
        print("AllowedClassMigration-[[ "+"TODO" + " ]] ")
        print("conversionOrder-[[ "+str(self.__conversionOrder)+ " ]] ")
        print("classAllocationOrder - [["+str(self.__classAllocationOrder)+ " ]] ")
        print("neighbour-[[ "+str(self.__neughbourl)+ " ]] ")
        print("modelformula-[[ "+str(self.__modelformula)+ " ]] ")
        print("outputfile-[[ "+self.__OutputFile+ " ]] ")
        print("suitabilityFileDirectory-[[ "+str(self.__suitabilityFileDirectory)+ " ]] ")

    def prepareClassAllocationOrder(self):
        #Prepare from migration order list
        #For the time being
        self.__classAllocationOrder=[]
        if(self.ui.cbclassallocation_DemandAllocation.isChecked()):
            for j in list(range(0,self.__noOfClasses,1)):
                self.__classAllocationOrder.append(int(str(self.ui.twPolicies_DemandAllocation.item(j,1).text())))
        else:
            self.__classAllocationOrder=None
            
    @pyqtSlot(bool)
    def on_cbEnable_SuitabilityMapGeneration_clicked(self,state):
        print('cbEnable_SuitabilityMapGeneration')
        self.ui.leSuitablityFile_OutputGenerationSuitabilityMapGeneration.setEnabled(state)
        self.ui.pbSelectFileSuitablityFile_OutputGenerationSuitabilityMapGeneration.setEnabled(state)
        self.ui.lbSuitablityfile.setEnabled(state)

    @pyqtSlot()
    def on_pbSelectFileSuitablityFile_OutputGenerationSuitabilityMapGeneration_clicked(self):
        self.__suitabilityFileDirectory = QFileDialog.getExistingDirectory(self, "Open suitablity Directory",".", QFileDialog.ShowDirsOnly);
        self.ui.leSuitablityFile_OutputGenerationSuitabilityMapGeneration.setText(str(self.__suitabilityFileDirectory))


    def prepareSuitabilityFileDirectory(self):
        name=str(self.ui.leSuitablityFile_OutputGenerationSuitabilityMapGeneration.text())
        if(len(name)!=0):
            self.__suitabilityFileDirectory=name
        else:
            self.__suitabilityFileDirectory=None

    def prepareReferenceFile(self):
        name=str(self.ui.leActualFile_AccuracyAssesment.text())
        if(len(name)!=0):
            self.__ReferenceFile=name
        else:
            self.__ReferenceFile=None

    def runRCommand(self, on_progress=None):
        """Pure-Python replacement for the original ``runRCommand``
        (R: ``genratePredictedMap``). Fixes a bug in the original: it
        always passed ``classAllocationOrder=R.NA_Logical``, silently
        discarding the value ``prepareClassAllocationOrder`` had just
        built from the GUI table — this port passes the real value.
        """
        return self.controller.run_prediction_from_gui_state(
            model_type=self.__modeltype,
            t1_file=self.__T0File,
            t2_file=self.__T1File,
            class_names=self.__className,
            t1_drivers=self.__DriverDictionaryT1,
            t2_drivers=self.__DriverDictionaryT2,
            na_value=self.__modelNAValue,
            demand=self.__demand,
            restrict_spatial_migration=self.__restricSpatial,
            neighbour=self.__neughbourl,
            output_file=str(self.__OutputFile),
            conversion_order=self.__conversionOrder,
            class_allocation_order=self.__classAllocationOrder,
            model_formula=self.__modelformula,
            mask_file=self.__MASKFile,
            aoi_file=self.__AOIFile,
            suitability_file_directory=self.__suitabilityFileDirectory,
            on_progress=on_progress,
        )

    @pyqtSlot()
    def on_pbExecute_DemandAllocation_clicked(self):
        print('pbExecute_DemandAllocation_clicked')
        # Find Accuracy stays enabled regardless of Execute's state — it
        # reads whichever Actual/Predicted/Base files are in its own
        # fields, independent of whether Execute has just been run.
        self.prepareDataForExecution()

        def on_success(result):
            path=os.path.join(str(self.__projectDirectory),self.currentLogTime+'-execute.log')
            with open(path, "w") as fh:
                fh.write(f"Output written to: {result.output_file}\n")
                if result.warnings:
                    fh.write("\nWarnings:\n" + "\n".join(result.warnings) + "\n")
            self.ui.tabWidget.setCurrentIndex(5)
            # NB: no hardcoded progressBar value here — the QtProgressRelay
            # wired up below already drove it to 100 ("Complete", the real
            # pipeline's own last report()) by the time this success
            # callback runs. A stray setProperty("value", 60) here used to
            # silently stomp that back down to a stale wizard-step milestone
            # right after Execute finished.
            self.ui.gbSummary_AccuracyAssesment.setEnabled(True)

            # View Maps' file selector was never populated with anything to
            # select — fill it with this run's actual outputs (predicted
            # map, any multi-step intermediates, per-class suitability
            # maps), current run replacing whatever a previous run left.
            self.ui.cbFile_ViewMaps.clear()
            self.ui.cbFile_ViewMaps.addItem(result.output_file)
            for step_file in result.step_files:
                self.ui.cbFile_ViewMaps.addItem(step_file)
            for suitability_file in result.suitability_files.values():
                self.ui.cbFile_ViewMaps.addItem(suitability_file)
            self.ui.cbFile_ViewMaps.setCurrentIndex(0)

        relay = QtProgressRelay()
        relay.progress_changed.connect(self._on_pipeline_progress)
        self._progress_relay = relay  # keep alive until the worker finishes

        self._run_in_background(
            self.runRCommand,
            on_success,
            busy_widgets=[self.ui.pbExecute_DemandAllocation],
            on_progress=relay.report,
        )
        
    def _buildScenarioFile(self):
        """Current GUI state -> a ScenarioFile (see LULC/scenario.py). Shared
        by Save (writes it) and could be reused wherever else the full
        current scenario needs exporting."""
        self.prepareDataForExecution()
        try:
            class_ids = self.controller.get_class_codes(self.__T0File, na_value=self.__modelNAValue)
        except PipelineError:
            class_ids = list(range(1, len(self.__className) + 1))

        model_types = self.__modeltype if isinstance(self.__modeltype, list) else [self.__modeltype] * len(self.__className)
        color_table = self.ui.twColorTable_ViewMaps
        classes = [
            ScenarioClass(
                name=self.__className[i],
                class_id=class_ids[i] if i < len(class_ids) else i + 1,
                model_type=model_types[i] if i < len(model_types) else "logistic",
                demand=(self.__demand[i] if self.__demand is not None else None),
                inertia=(self.__restricSpatial[i] if self.__restricSpatial is not None else None),
                legend_text=(color_table.item(i, 1).text() if i < color_table.rowCount() else None),
                colour=(color_table.item(i, 2).background().color().name() if i < color_table.rowCount() else None),
            )
            for i in range(len(self.__className))
        ]

        neighbour = self.__neughbourl or []
        return ScenarioFile(
            t1_file=self.__T0File,
            t1_year=self.__T0Year or None,
            t2_file=self.__T1File,
            t2_year=self.__T1Year or None,
            output_file=str(self.__OutputFile),
            na_value=self.__modelNAValue,
            area_of_interest_file=self.__AOIFile,
            mask_file=self.__MASKFile,
            drivers_t1=dict(self.__DriverDictionaryT1),
            drivers_t2=dict(self.__DriverDictionaryT2),
            classes=classes,
            class_allocation_order=self.__classAllocationOrder,
            conversion_order=self.__conversionOrder,
            spatial_context=ScenarioSpatialContext(
                enabled=self.__neughbourl is not None,
                window_size=int(neighbour[0]) if len(neighbour) > 0 else 3,
                steps=int(neighbour[1]) if len(neighbour) > 1 else 1,
                write_step_output=bool(neighbour[2]) if len(neighbour) > 2 else False,
            ),
            suitability_file_directory=self.__suitabilityFileDirectory,
            accuracy_assessment=ScenarioAccuracyAssessment(
                reference_file=self.__ReferenceFile,
                predicted_file=(self.ui.lePredictedFile_AccuracyAssesment.text().strip() or None),
                base_file=(self.ui.leBaseFile_AccuracyAssesment.text().strip() or None),
                display_mode=(
                    "overall" if self.ui.rbAgreementIndexOverall_AccuracyAssesmentDetailed.isChecked() else "classwise"
                ),
            ),
            map_composition=ScenarioMapComposition(
                source_file=(self.ui.cbFile_ViewMaps.currentText() or None),
                title=self.ui.leTitle_ViewMaps.text(),
                legend_heading=self.ui.leLegendHeading_ViewMaps.text(),
            ),
        )

    @pyqtSlot()
    def on_actionSave_File_triggered(self):
        """Write the current scenario to a YAML "parameters file"
        — replaces the original's standalone-script export (a runnable .R
        script in the R version, a runnable .py script in this port's first
        pass) with something Open File can actually read back in, and that
        ``OpenLDM.py --config`` can also run headlessly. Any GUI checkbox left
        unchecked (Demand/Class Inertia/Allocation Order/Spatial Context)
        writes as an explicit YAML ``null`` "placeholder" — resolved to a
        real default on load, see LULC/scenario.py."""
        print("save file clicked")
        scenario = self._buildScenarioFile()
        fileLocation = QFileDialog.getSaveFileName(self, "Save File",
                                                        self.__currentDirectory,"YAML (*.yaml *.yml)")[0];
        if not fileLocation:
            return
        (dirName, fileName) = os.path.split(str(fileLocation))
        fileType = splitext(fileName)[1]
        mysavefile = str(fileLocation) if fileType in (".yaml", ".yml") else str(fileLocation) + ".yaml"
        scenario.to_yaml(mysavefile)

    @pyqtSlot()
    def on_actionExit_triggered(self):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QtWidgets.QMessageBox.question(self, 'OpenLDM Message',
                                                   quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)
        if reply == QtWidgets.QMessageBox.Yes:
            self.statusMessage.stop()
            self.statusMessage.wait(2000)
            log_bridge.detach(self._log_handler)
            if self.__embedded:
                # Embedded in a host application (e.g. the QGIS plugin) that
                # owns the QApplication -- close only this window, not the
                # whole host. closeEvent() is already safe to call this way
                # (no QApplication.quit() in it).
                self.close()
            else:
                QtWidgets.QApplication.quit()

    @pyqtSlot()
    def on_actionOpen_File_triggered(self):
        """Load a YAML scenario file and populate every tab's
        fields/tables from it — Data Preparation, Driver Selection T0/T1,
        Demand Allocation, Accuracy Assessment, View Maps — without
        running any compute step. "View Model Statistics", Execute, Show
        and Export are never auto-triggered here; the user reviews and
        runs those themselves when ready. Validate *is* triggered (Data
        Preparation's own consistency check) — nothing past it can be
        meaningfully populated without first knowing the files are valid
        rasters.

        The later tabs' tables (Model Analysis, Driver Selection T1,
        Demand Allocation) are normally only built by
        on_pbNext_DriverSelectionT0_clicked, which itself only runs after
        a real fit — its createLULCVsDriverCoefficientMatrix() indexes
        into self.modelSummary per class and would IndexError against an
        empty dict. A placeholder self.modelSummary (one blank entry per
        class) makes that method safe to call directly without a real
        fit — breakCoeffDetails already degrades a tab-less string to
        "NA" placeholders in every cell (see its own docstring), so the
        coefficient table just shows blanks until the user runs a real
        fit themselves."""
        fileLocation, _ = QFileDialog.getOpenFileName(
            self, "Open File", self.__currentDirectory, "YAML (*.yaml *.yml)"
        )
        if not fileLocation:
            return
        self.loadScenarioFile(fileLocation)

    def loadScenarioFile(self, fileLocation):
        """The actual load-and-populate logic behind File > Open, factored
        out so ``OpenLDM.py --mode gui --config <path>`` can call it
        directly (right after the window is shown) without going through
        a file-picker dialog first. A malformed/corrupted file shows the
        same ``QMessageBox.critical`` either way — the GUI still opens and
        stays usable, it just doesn't have a scenario loaded into it."""
        try:
            scenario = ScenarioFile.from_yaml(fileLocation)
        except ScenarioFileError as exc:
            QMessageBox.critical(self, "OpenLDM Error", f"Could not load {fileLocation}: {exc}")
            return

        self.ui.leT0File_DataPreparationInputSectionDataInput.setText(scenario.t1_file or "")
        self.on_leT0File_DataPreparationInputSectionDataInput_editingFinished()
        self.ui.leT1File_DataPreparationInputSectionDataInput.setText(scenario.t2_file or "")
        self.on_leT1File_DataPreparationInputSectionDataInput_editingFinished()
        self.ui.leOutputFile_DataPreparationOutputSection.setText(scenario.output_file or "")
        self.on_leOutputFile_DataPreparationOutputSection_editingFinished()
        if scenario.t1_year:
            self.ui.leT0Year_DataPreparationInputSectionDataInput.setText(str(scenario.t1_year))
            self.on_leT0Year_DataPreparationInputSectionDataInput_editingFinished()
        if scenario.t2_year:
            self.ui.leT1Year_DataPreparationInputSectionDataInput.setText(str(scenario.t2_year))
            self.on_leT1Year_DataPreparationInputSectionDataInput_editingFinished()
        if scenario.accuracy_assessment.reference_file:
            self.ui.leActualFile_AccuracyAssesment.setText(scenario.accuracy_assessment.reference_file)
        if scenario.accuracy_assessment.base_file:
            self.ui.leBaseFile_AccuracyAssesment.setText(scenario.accuracy_assessment.base_file)
        if scenario.accuracy_assessment.display_mode == "overall":
            self.ui.rbAgreementIndexOverall_AccuracyAssesmentDetailed.setChecked(True)
        else:
            self.ui.rbAgreementIndexClasswise_AccuracyAssesmentDetailed.setChecked(True)
        if scenario.map_composition.title:
            self.ui.leTitle_ViewMaps.setText(scenario.map_composition.title)
        if scenario.map_composition.legend_heading:
            self.ui.leLegendHeading_ViewMaps.setText(scenario.map_composition.legend_heading)
        if scenario.accuracy_assessment.predicted_file:
            # Overrides the auto-sync the Output File field above just did
            # (on_leOutputFile_..._editingFinished also sets this field to
            # match) -- only takes effect when the scenario explicitly
            # points Accuracy Assessment at a *different* raster.
            self.ui.lePredictedFile_AccuracyAssesment.setText(scenario.accuracy_assessment.predicted_file)
        if scenario.mask_file:
            self.ui.leMask_DataPreparationInputSectionDataInput.setText(scenario.mask_file)
        if scenario.area_of_interest_file:
            self.ui.leAreaOfInterest_DataPreparationInputSectionDataInput.setText(scenario.area_of_interest_file)
        if scenario.na_value is not None:
            self.ui.leNAValue_DataPreparationInputSectionProjectSection.setText(str(scenario.na_value))

        if not self.ui.pbValidate_DataPreparation.isEnabled():
            QMessageBox.critical(
                self, "OpenLDM Error",
                "Scenario is missing required Data Preparation fields "
                "(T1/T2 file, output file, or T1/T2 year).",
            )
            return
        self.on_pbValidate_DataPreparation_clicked()
        if not self.ui.pbNextDataPreparation.isEnabled():
            return  # Validate already showed the dataset issues.
        self.on_pbNextDataPreparation_clicked()

        for name, path in scenario.drivers_t1.items():
            self.on_pbAddDriver_DriverSelectionT0SelectDrivers_clicked()
            row = self.ui.twSelectDrivers_DriverSelectionT0.rowCount() - 1
            self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(row, 0).setText(name)
            self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(row, 1).setText(path)

        model_types = {c.model_type for c in scenario.classes} if scenario.classes else {"logistic"}
        uniform_model_type = next(iter(model_types)) if len(model_types) == 1 else "logistic"
        radio = {
            "logistic": self.ui.rbLogisticRegression_DriverSelectionT0DoModelFitting,
            "regression": self.ui.rbLinearRegression_DriverSelectionT0DoModelFitting,
            "nnet": self.ui.rbNeuralregression_DriverSelectionT0DoModelFitting,
            "randomForest": self.ui.rbRandomForest_DriverSelectionT0DoModelFitting,
            "svm": self.ui.rbSVM_DriverSelectionT0DoModelFitting,
        }.get(uniform_model_type, self.ui.rbLogisticRegression_DriverSelectionT0DoModelFitting)
        radio.setChecked(True)

        # No real fit -- seed a placeholder modelSummary (see docstring)
        # so the later tables can still be built. createModelStatisticDetails()
        # re-syncs __T0File/__T1File/__DriverDictionaryT1/NA value/AOI/Mask
        # from the fields above, same as a real "View Model Statistics"
        # click would before fitting.
        self.createModelStatisticDetails()
        try:
            class_ids = self.controller.get_class_codes(self.__T0File, na_value=self.__modelNAValue)
        except PipelineError as exc:
            QMessageBox.critical(self, "OpenLDM Error", str(exc))
            return
        self.modelSummary = {f"placeholder{i}": "" for i in range(len(class_ids))}

        self._continueOpenScenario(scenario)

    __MODEL_TYPE_LABELS = {
        "logistic": "Logistic Regression",
        "regression": "Linear Regression",
        "nnet": "Neural Regression",
        "randomForest": "Random Forest",
        "svm": "SVM",
    }

    def _continueOpenScenario(self, scenario):
        """Continuation of on_actionOpen_File_triggered — builds the
        remaining 4 tabs' state from the loaded scenario. No compute step
        runs here; on_pbNext_DriverSelectionT0_clicked/on_pbNext_
        ModelAnalysis_clicked/on_pbNext_DriverSelectionT1_clicked are cheap
        table-building/tab-switch UI actions, not pipeline runs."""
        self.on_pbNext_DriverSelectionT0_clicked()

        color_table = self.ui.twColorTable_ViewMaps
        for i, cls in enumerate(scenario.classes):
            if cls.name and i < len(self.__className):
                self._sync_class_name(i, cls.name)
            if i >= color_table.rowCount():
                continue
            if cls.legend_text:
                color_table.item(i, 1).setText(cls.legend_text)
            # Every class gets a colour, not just ones the scenario set
            # explicitly -- cls.colour or the auto-generated greyscale
            # ramp buildtwColorTable_ViewMaps already assigned each row,
            # same as a freshly-built, never-edited color table.
            colour = cls.colour or color_table.item(i, 2).background().color().name()
            color_table.item(i, 2).setBackground(QBrush(QColor(colour)))

        if isinstance(scenario.conversion_order, (list, tuple)):
            # An explicit N x N migration-order matrix (the "User Defined"
            # alternative to the "TP"/system-computed default) -- Save
            # already captures this correctly (prepareConversionOrder()
            # reads it straight from this same table), but Open never
            # restored it back in, silently discarding a loaded scenario's
            # custom matrix in favor of whatever "TP" default happened to
            # be showing. Checking the radio rebuilds the table (from the
            # already-synced self.__className, so names aren't lost) and
            # enables it; the matrix values are then written directly.
            self.ui.rbUserDefined_ModelAnalysisMigrationOrder.setChecked(True)
            table = self.ui.twMigrationOrder_ModelAnalysis
            matrix = scenario.conversion_order
            for row in range(min(len(matrix), self.__noOfClasses)):
                row_values = matrix[row]
                for col in range(min(len(row_values), self.__noOfClasses)):
                    item = table.item(row, col + 1)
                    if item is not None:
                        item.setText(str(row_values[col]))

        self.on_pbNext_ModelAnalysis_clicked()

        driver_names = list(self.__DriverDictionaryT1.keys())
        table = self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1
        for col_idx, name in enumerate(driver_names):
            path95 = scenario.drivers_t2.get(name)
            if not path95:
                continue
            col = 3 + col_idx
            for row in range(table.rowCount()):
                item = table.item(row, col)
                if item is not None:
                    item.setText(path95)

        model_types = [c.model_type for c in scenario.classes]
        if len(set(model_types)) > 1:
            self.ui.rbIndvidualSelection_DriverSelectionT1SelectModelTypeAndDrivers.setChecked(True)
            for row, model_type in enumerate(model_types):
                combo = table.cellWidget(row, 2)
                label = self.__MODEL_TYPE_LABELS.get(model_type)
                if combo is not None and label is not None:
                    combo.setCurrentText(label)

        self.on_pbNext_DriverSelectionT1_clicked()

        if any(c.demand is not None for c in scenario.classes):
            self.ui.cbUserDefinedDemand_DemandAllocation.setChecked(True)
            self.on_cbUserDefinedDemand_DemandAllocation_clicked(True)
            for i, c in enumerate(scenario.classes):
                if i < self.__noOfClasses:
                    self.ui.twPolicies_DemandAllocation.item(i, 2).setText(str(c.demand if c.demand is not None else 0))

        if any(c.inertia is not None for c in scenario.classes):
            self.ui.cbUserDefinedClassInertia_DemandAllocation.setChecked(True)
            self.on_cbUserDefinedClassInertia_DemandAllocation_clicked(True)
            for i, c in enumerate(scenario.classes):
                if i < self.__noOfClasses:
                    self.ui.twPolicies_DemandAllocation.item(i, 3).setText(str(c.inertia if c.inertia is not None else 0.0))

        if scenario.class_allocation_order:
            self.ui.cbclassallocation_DemandAllocation.setChecked(True)
            self.on_cbclassallocation_DemandAllocation_clicked(True)
            for i, order in enumerate(scenario.class_allocation_order):
                if i < self.__noOfClasses:
                    self.ui.twPolicies_DemandAllocation.item(i, 1).setText(str(order))

        if scenario.spatial_context.enabled:
            self.ui.cbEnable_DemandAllocationSpatialContext.setChecked(True)
            self.on_cbEnable_DemandAllocationSpatialContext_clicked(True)
            for combo, value in (
                (self.ui.cbWindowSize_DemandAllocationSpatialContext, str(scenario.spatial_context.window_size)),
                (self.ui.cbInSteps_DemandAllocationSpatialContext, str(scenario.spatial_context.steps)),
                (self.ui.cbStepOutputRequired_DemandAllocationSpatilaContext,
                 "Yes" if scenario.spatial_context.write_step_output else "No"),
            ):
                idx = combo.findText(value)
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        self.ui.tabWidget.setCurrentIndex(4)
        QMessageBox.information(
            self, "Scenario loaded",
            "Scenario loaded — all tabs are populated, but no computation has "
            "run yet. Review the tabs, then click \"View Model Statistics\" "
            "(Driver Selection) and \"Execute\" (Demand Allocation) yourself "
            "when ready.",
        )

    def calculateStatistics(self):
        """Pure-Python replacement for the original's createTM/kappa/
        PyKappasummary chain + '~~'-delimited string re-parsing.
        LULCAlgorithms.get_kappa_summary already returns a structured
        accuracy.KappaStatistics object, so this reads fields directly
        off it instead of round-tripping through a string."""
        print('kappa')
        actualFile=str(self.ui.leActualFile_AccuracyAssesment.text().strip())
        predictedFile=str(self.ui.lePredictedFile_AccuracyAssesment.text().strip())
        result = self.controller.get_kappa_summary(actualFile, predictedFile, na_value=self.__modelNAValue)
        stats = result["statistics"]
        cm = result["confusion_matrix"]

        path=os.path.join(str(self.__projectDirectory),self.currentLogTime+'-accuracy.log')
        with open(path, "w") as fh:
            fh.write(str(cm) + "\n\n" + stats.summary_text())

        acc_lo95, acc_hi95 = stats.confidence_interval("accuracy", 0.05)
        acc_lo99, acc_hi99 = stats.confidence_interval("accuracy", 0.01)
        kap_lo95, kap_hi95 = stats.confidence_interval("kappa", 0.05)
        kap_lo99, kap_hi99 = stats.confidence_interval("kappa", 0.01)

        self.__noOfObservation=f"{stats.sum_n:.0f}"
        self.__overallOfAccuracy=f"{stats.sum_naive:.4f}"
        self.__overallOfAccuracyCI95=f"{acc_lo95:.4f}-{acc_hi95:.4f}"
        self.__overallOfAccuracyCI99=f"{acc_lo99:.4f}-{acc_hi99:.4f}"
        self.__userAccuracy=[f"{v:.4f}" for v in stats.user_naive]
        self.__producerReliability=[f"{v:.4f}" for v in stats.prod_naive]
        self.__overallKappa=f"{stats.sum_kappa:.4f}"
        self.__overallKappaCI95=f"{kap_lo95:.4f}-{kap_hi95:.4f}"
        self.__overallKappaCI99=f"{kap_lo99:.4f}-{kap_hi99:.4f}"

        noOfClasses=cm.shape[0]
        self.__transitionMatrix = [[0 for i in list(range(0,noOfClasses+1,1))] for j in range(0,noOfClasses+1,1)]
        for i in list(range(0,noOfClasses,1)):
            for j in list(range(0,noOfClasses,1)):
                value=int(cm[j][i])
                self.__transitionMatrix[j][i]=value
                self.__transitionMatrix[noOfClasses][i]+=value
                self.__transitionMatrix[j][noOfClasses]+=value
                self.__transitionMatrix[noOfClasses][noOfClasses]+=value

        # Pontius agreement-index decomposition (R: kappa.agreementindex).
        # "Simulation" = the model's predicted map, "actual" =
        # the reference map it's checked against — same two files as the
        # kappa chain above, matching R's own naming. Base file is optional
        # (leBaseFile_AccuracyAssesment); when blank, ksimulation/
        # ktransition/ktranslocation come back NaN (see pontius.py) and are
        # left blank in the detail table instead of computed.
        baseFile = str(self.ui.leBaseFile_AccuracyAssesment.text().strip()) or None
        self.__pontiusResult = self.controller.kappa_agreement_index(
            predictedFile, actualFile, baseFile, na_value=self.__modelNAValue,
        )
    @pyqtSlot(bool)
    def on_rbConfusionMatrix_AccuracyAssesmentDetailed_toggled(self,checked):
        if(checked):
            self.buildtwDetailed_AccuracyAssesment(self.ui.twDetailed_AccuracyAssesment,True)
            for i in list(range(0,len(self.__transitionMatrix),1)):
                for j in list(range(0,len(self.__transitionMatrix),1)):
                    self.ui.twDetailed_AccuracyAssesment.item(i,j).setText(str(int(self.__transitionMatrix[i][j])))
    @pyqtSlot(bool)
    def on_rbPreditedMapAccuracy_AccuracyAssesmentDetailed_toggled(self,checked):
        if(checked):
            self.buildtwDetailed_AccuracyAssesment(self.ui.twDetailed_AccuracyAssesment,False)
            for j in list(range(0,len(self.__userAccuracy),1)):
                self.ui.twDetailed_AccuracyAssesment.item(j,0).setText(str(self.__userAccuracy[j]))
    @pyqtSlot(bool)
    def on_rbReferenceMapReliablity_AccuracyAssesmentDetailed_toggled(self,checked):
        if(checked):
            self.buildtwDetailed_AccuracyAssesment(self.ui.twDetailed_AccuracyAssesment,False)
            for j in list(range(0,len(self.__producerReliability),1)):
                self.ui.twDetailed_AccuracyAssesment.item(j,0).setText(str(self.__producerReliability[j]))

    # Row order for the Pontius agreement-index tables below — matches R's
    # summary.kappa.agreementindex.tabel, which likewise excludes
    # klocation and the raw disagreement-decomposition fields (A/C/Q/E/S/D)
    # from this particular table.
    __AGREEMENT_INDEX_METRICS = (
        "kstandard", "kno", "kallocation", "khistogram",
        "kquantity", "ksimulation", "ktransition", "ktranslocation",
    )

    __AGREEMENT_INDEX_BASE_DEPENDENT = ("ksimulation", "ktransition", "ktranslocation")

    def _active_agreement_index_metrics(self):
        """The 8 metrics, minus the 3 that need a base file
        (ksimulation/ktransition/ktranslocation) when none was given — those
        rows are omitted from the table entirely rather than shown blank."""
        result = self.__pontiusResult
        has_base = not isnan(result.ksimulation_overall)
        if has_base:
            return self.__AGREEMENT_INDEX_METRICS
        return tuple(
            name for name in self.__AGREEMENT_INDEX_METRICS
            if name not in self.__AGREEMENT_INDEX_BASE_DEPENDENT
        )

    def buildtwDetailed_AccuracyAssesment_AgreementIndex(self, table, classwise, metric_names):
        """Pontius agreement-index table shape — rows are always
        metric names; columns are per-class when ``classwise`` else a
        single "Overall" column. Mirrors buildtwDetailed_AccuracyAssesment's
        blank-non-editable-item + header-label pattern, just transposed
        (that helper always puts classes on rows)."""
        # "kstandard" -> "Kstandard", matching R's metric<-c("Kstandard",...) labels.
        metric_labels = [name.capitalize() for name in metric_names]
        if classwise:
            class_labels = list(self.__className) if self.__className else [
                f"Class{i + 1}" for i in range(self.__pontiusResult.no_of_class)
            ]
            table.setColumnCount(len(class_labels))
        else:
            class_labels = ["Overall"]
            table.setColumnCount(1)
        table.setRowCount(len(metric_labels))

        row = table.rowCount(); col = table.columnCount()
        for i in list(range(0, row, 1)):
            for j in list(range(0, col, 1)):
                item = QTableWidgetItem()
                item.setFlags(item.flags()^QtCore.Qt.ItemIsEditable)
                table.setItem(i, j, item)
        table.setVerticalHeaderLabels(metric_labels)
        table.setHorizontalHeaderLabels(class_labels)

    def _populate_agreement_index_table(self, classwise):
        result = self.__pontiusResult
        if result is None:
            # Bug: toggling Classwise/Overall before "Find Accuracy" has
            # ever run (self.__pontiusResult isn't computed until
            # calculateStatistics()) crashed here — nothing to show yet,
            # so just leave the table as-is instead.
            return
        metric_names = self._active_agreement_index_metrics()
        self.buildtwDetailed_AccuracyAssesment_AgreementIndex(
            self.ui.twDetailed_AccuracyAssesment, classwise, metric_names,
        )
        for i, name in enumerate(metric_names):
            values = getattr(result, f"{name}_classwise") if classwise else [getattr(result, f"{name}_overall")]
            for j, v in enumerate(values):
                if isnan(v):
                    continue
                self.ui.twDetailed_AccuracyAssesment.item(i, j).setText(f"{float(v):.7f}")

    @pyqtSlot(bool)
    def on_rbAgreementIndexClasswise_AccuracyAssesmentDetailed_toggled(self, checked):
        if(checked):
            self._populate_agreement_index_table(classwise=True)

    @pyqtSlot(bool)
    def on_rbAgreementIndexOverall_AccuracyAssesmentDetailed_toggled(self, checked):
        if(checked):
            self._populate_agreement_index_table(classwise=False)

    def populateDataIntoTableDetailed_AccuracyAssesment(self):
        self.calculateStatistics()
        self.ui.leOverallKappa_AccuracyAssesment.setText(self.__overallKappa)
        self.ui.leOverallKappaCI_AccuracyAssesment95.setText(self.__overallKappaCI95)
        self.ui.leOverallKappaCI_AccuracyAssesment99.setText(self.__overallKappaCI99)
        self.ui.leOverallAccuracy_AccuracyAssesment.setText(self.__overallOfAccuracy)
        self.ui.leOverallAccuracyCI_AccuracyAssesment95.setText(self.__overallOfAccuracyCI95)
        self.ui.leOverallAccuracyCI_AccuracyAssesment99.setText(self.__overallOfAccuracyCI99)
        self.ui.leNoOfObservation_AccuracyAssesmentSummary.setText(self.__noOfObservation)
        
    @pyqtSlot(int)
    def on_cbStepOutputRequired_DemandAllocationSpatilaContext_currentIndexChanged(self,index):
        print('StepOutputRequired_DemandAllocationSpatialContext'+str(index))

    @pyqtSlot(int)
    def on_cbInSteps_DemandAllocationSpatialContext_currentIndexChanged(self,index):
        print('cbInSteps_DemandAllocationSpatialContext'+str(index))

    @pyqtSlot(int)
    def on_cbWindowSize_DemandAllocationSpatialContext_currentIndexChanged(self,index):
        print('WindowSize_DemandAllocationSpatialContext'+str(index))

    @pyqtSlot()
    def on_pbSelectFileActualFile_AccuracyAssesment_clicked(self):
        print('pbSelectFileActualFile_AccuracyAssesment')
        fileLocation,_ = QFileDialog.getOpenFileName(self, "Open File",self.__currentDirectory,"Raster (*.tif *.img)")
        self.ui.leActualFile_AccuracyAssesment.setText(str(fileLocation).strip())

    @pyqtSlot()
    def on_pbSelectFilePredictedFile_AccuracyAssesment_clicked(self):
        print('pbSelectFilePredictedFile_AccuracyAssesment')
        fileLocation,_ = QFileDialog.getOpenFileName(self, "Open File",self.__currentDirectory,"Raster (*.tif *.img)")
        self.ui.lePredictedFile_AccuracyAssesment.setText(str(fileLocation).strip())

    @pyqtSlot()
    def on_pbSelectFileBaseFile_AccuracyAssesment_clicked(self):
        """Optional third raster for the Pontius agreement-index
        decomposition (R: kappa.agreementindex's baseFile)."""
        print('pbSelectFileBaseFile_AccuracyAssesment')
        fileLocation,_ = QFileDialog.getOpenFileName(self, "Open File",self.__currentDirectory,"Raster (*.tif *.img)")
        self.ui.leBaseFile_AccuracyAssesment.setText(str(fileLocation).strip())

    @pyqtSlot()
    def on_pbFindAccuracy_AccuracyAssesment_clicked(self):
        print('pbFindAccuracy_AccuracyAssesment')
        try:
            self.populateDataIntoTableDetailed_AccuracyAssesment()
        except PipelineError as exc:
            QMessageBox.critical(self, "OpenLDM Error", str(exc))
            return
        self.ui.gbDetailed_AccuracyAssesment.setEnabled(True)
        self.ui.gbSummarStats_AccuracyAssesementSummary.setEnabled(True)

    @pyqtSlot(bool)
    def on_rbAuto_ModelAnalysisMigrationOrder_toggled(self,checked):
        if(checked):
            self.ui.twMigrationOrder_ModelAnalysis.setEnabled(False)
            self.__conversionOrder='TP'

    @pyqtSlot(bool)
    def on_rbLogisiticRegression_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setEnabled(False)

    @pyqtSlot(bool)
    def on_rbLinearRegression_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setEnabled(False)

    @pyqtSlot(bool)
    def on_rbNeuralRegression_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setEnabled(False)

    def on_rbRandomForest_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setEnabled(False)

    def setModelAnalysisModelType(self):
        if(self.__modeltype == "logistic" ):
            self.ui.rbLogisiticRegression_ModelAnalysisViewModelCoeeffcient.setChecked(True)
        elif(self.__modeltype == "svm" ):
            self.ui.rbSVM_ModelAnalysisViewModelCoeeffcient.setChecked(True)
            
    @pyqtSlot(int,int)
    def on_twViewModelCoefficint_ModelAnalysis_cellClicked(self,row,col):
        item=self.ui.twViewModelCoefficint_ModelAnalysis.item(row,col)
        if(col>2):
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.item(row,col).setCheckState(item.checkState())


    def makeCombo(self):
        cbModel = QComboBox()
        cbModel.setMaxCount(5)
        cbModel.setObjectName("cbModel"+str(self.ui.twSelectDrivers_DriverSelectionT0.rowCount()-1))
        cbModel.addItem("iLoR")
        cbModel.addItem("iLiR")
        cbModel.addItem("iMLP")
        cbModel.addItem("iMRF")
        cbModel.addItem("iSVM")
        cbModel.setItemText(0, "Logistic Regression")
        cbModel.setItemText(1, "Linear Regression")
        cbModel.setItemText(2, "Neural Regression")
        cbModel.setItemText(3, "Random Forest")
        cbModel.setItemText(4, "SVM")
        return(cbModel)

    def populateDataIntoTableModelTypeAndDriversDriverSelectionT1(self):
        #Populate the DriversFile in DriverS electionT1 tab
        for j in list(range(0,self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount(),1)):
            item1 = self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.item(j,1)
            item1.setText(self.__className[j])
        for j in list(range(0,self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount(),1)):
            for k in list(range(3,self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.columnCount(),1)): #First three colums are reserved for classname,number and modetype
                item1 = self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.item(j,k)
                item1.setText(self.__driversT1[k-3]) #The Location of Drives File
                toolstr = str(self.__confidenceinterval[j][k-2][1])+" "+str(self.__confidenceinterval[j][k-2][2]) #k-3+1 since Intercept is at k-3
                #item1.setToolTip(QtGui.QApplication.translate("LULCModel", toolstr, None, QtGui.QApplication.UnicodeUTF8))
                item1.setToolTip(toolstr)

    @pyqtSlot()
    def on_Suitable_clicked(self):
        self.ui.tabWidget.setCurrentIndex(3)
        self.ui.progressBar.setProperty("value",40)


    def toolTip(self, final, m, n):
        self.item = self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.item(m, n)
        self.Coeffcient.append(self.__confidenceinterval[n-2][0])
        toolstr = str(self.__confidenceinterval[n-2][0])+" "+str(self.checkStar(self.__confidenceinterval[n-2]))
        self.item.setToolTip(QtGui.QApplication.translate("LULCModel", toolstr, None, QtGui.QApplication.UnicodeUTF8))
        return

    @pyqtSlot()
    def on_pbNextDemandAlloc_clicked(self):
        self.ui.twDemandAlloc.setRowCount(self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount())
        self.ui.tabWidget.setCurrentIndex(4)
        self.ui.progressBar.setProperty("value",50)


    def fill_table_with_label(this,thistable,labelText):
        i=0
        while i<thistable.rowCount():
            label = QLabel(labelText)
            thistable.setCellWidget(i, 3,label)
            i=i+1

    @pyqtSlot(bool)
    def on_rbLogisticRegression_DriverSelectionT0DoModelFitting_toggled(self,checked):
        if(checked):
            self.__modeltype = "logistic"
            self.ui.pbViewModelStatistics_DriverSelectionT0DoModelFitting.setEnabled(True)
            self.ui.rbLogisiticRegression_ModelAnalysisViewModelCoeeffcient.setChecked(True)
            self.ui.rbLogisiticRegression_DriverSelectionT1SelectModelTypeAndDrivers.setChecked(True)

    @pyqtSlot(bool)
    def on_rbLinearRegression_DriverSelectionT0DoModelFitting_toggled(self,checked):
        if(checked):
            self.__modeltype = "regression"
            self.ui.pbViewModelStatistics_DriverSelectionT0DoModelFitting.setEnabled(True)
            self.ui.rbLinearRegression_ModelAnalysisViewModelCoeeffcient.setChecked(True)
            self.ui.rbLinearRegression_DriverSelectionT1SelectModelTypeAndDrivers.setChecked(True)
        #fill_table_with_label(self.ui.twSelectDrivers_DriverSelectionT0,"Linear R")

    @pyqtSlot(bool)
    def on_rbNeuralregression_DriverSelectionT0DoModelFitting_toggled(self,checked):
        if(checked):
            self.__modeltype = "nnet"
            self.ui.pbViewModelStatistics_DriverSelectionT0DoModelFitting.setEnabled(True)
            self.ui.rbNeuralRegression_ModelAnalysisViewModelCoeeffcient.setChecked(True)
            self.ui.rbNeuralRegression_DriverSelectionT1SelectModelTypeAndDrivers.setChecked(True)
        #fill_table_with_label(self.ui.twSelectDrivers_DriverSelectionT0,"Neural Network")

    @pyqtSlot(bool)
    def on_rbRandomForest_DriverSelectionT0DoModelFitting_toggled(self,checked):
        if(checked):
            self.__modeltype = "randomForest"
            self.ui.pbViewModelStatistics_DriverSelectionT0DoModelFitting.setEnabled(True)
            self.ui.rbRandomForest_ModelAnalysisViewModelCoeeffcient.setChecked(True)
            self.ui.rbRandomForest_DriverSelectionT1SelectModelTypeAndDrivers.setChecked(True)
        #fill_table_with_label(self.ui.twSelectDrivers_DriverSelectionT0,"Random Forest")

    @pyqtSlot(bool)
    def on_rbSVM_DriverSelectionT0DoModelFitting_toggled(self,checked):
        if(checked):
            self.__modeltype = "svm"
            self.ui.pbViewModelStatistics_DriverSelectionT0DoModelFitting.setEnabled(True)
            self.ui.rbSVM_ModelAnalysisViewModelCoeeffcient.setChecked(True)
            self.ui.rbSVM_DriverSelectionT1SelectModelTypeAndDrivers.setChecked(True)

    @pyqtSlot(bool)
    def on_rbIndvidualSelection_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            self.__modeltype = "individual"
            i=0
            while i<self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount():
                combo=self.makeCombo()
                self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setCellWidget(i, 2,combo)
                i=i+1
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.resizeColumnsToContents()

    @pyqtSlot(bool)
    def on_rbLogisiticRegression_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            i=0
            while i<self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount():
                label=QLabel("Logistic Regression")
                self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setCellWidget(i, 2,label)
                i=i+1
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.resizeColumnsToContents()

    @pyqtSlot(bool)
    def on_rbLinearRegression_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            i=0
            while i<self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount():
                label=QLabel("Linear Regression")
                self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setCellWidget(i, 2,label)
                i=i+1
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.resizeColumnsToContents()

    @pyqtSlot(bool)
    def on_rbNeuralRegression_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            i=0
            while i<self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount():
                label=QLabel("Neural Regression")
                self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setCellWidget(i, 2,label)
                i=i+1
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.resizeColumnsToContents()

    @pyqtSlot(bool)
    def on_rbRandomForest_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            i=0
            while i<self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount():
                label=QLabel("Random Forest")
                self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setCellWidget(i, 2,label)
                i=i+1
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.resizeColumnsToContents()

    @pyqtSlot(bool)
    def on_rbSVM_DriverSelectionT1SelectModelTypeAndDrivers_toggled(self,checked):
        if(checked):
            i=0
            while i<self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount():
                label=QLabel("SVM")
                self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.setCellWidget(i, 2,label)
                i=i+1
            self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.resizeColumnsToContents()

    @pyqtSlot(int,int)
    def on_twSelectModelTypeAndDrivers_DriverSelectionT1_cellDoubleClicked(self,row,col):
        if(col>2):
            file,_ = QFileDialog.getOpenFileName(self, "Open File",self.__currentDirectory,"Raster (*.tif *.img)");
            (dirName, fileName) = os.path.split(str(file));
            self.__currentDirectory=dirName
            if((str(file))):
                i=0;
                while i<self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.rowCount():
                    item=self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.item(i,col)
                    item.setText(str(file))
                    i=i+1;
        self.ui.twSelectModelTypeAndDrivers_DriverSelectionT1.resizeColumnsToContents()

    @pyqtSlot()
    def on_leT0File_DataPreparationInputSectionDataInput_editingFinished(self):
        if(self.ui.leT0File_DataPreparationInputSectionDataInput.isModified()):
            self.ui.leT0File_DataPreparationInputSectionDataInput.setModified(False)
        if(not len(self.ui.leT0File_DataPreparationInputSectionDataInput.text())) == 0:
            self.__T0File=str(self.ui.leT0File_DataPreparationInputSectionDataInput.text().strip())
            self.enable_ValidateDataPreparation()

    @pyqtSlot()
    def on_leT1File_DataPreparationInputSectionDataInput_editingFinished(self):
        if(self.ui.leT1File_DataPreparationInputSectionDataInput.isModified()):
            self.ui.leT1File_DataPreparationInputSectionDataInput.setModified(False)
        if(not len(self.ui.leT1File_DataPreparationInputSectionDataInput.text())) == 0:
            self.__T1File=str(self.ui.leT1File_DataPreparationInputSectionDataInput.text().strip())
            self.enable_ValidateDataPreparation()

    @pyqtSlot()
    def on_leOutputFile_DataPreparationOutputSection_editingFinished(self):
        if(self.ui.leOutputFile_DataPreparationOutputSection.isModified()):
            self.ui.leOutputFile_DataPreparationOutputSection.setModified(False)
        if(not len(self.ui.leOutputFile_DataPreparationOutputSection.text())) == 0:
            outputText = self.ui.leOutputFile_DataPreparationOutputSection.text().strip()
            # Bug fix: this fires whenever the field loses focus, including
            # right after "Select File" has already put an absolute path in
            # it (on_pbSelectFileOutputFile_..._clicked) — blindly
            # prepending __currentDirectory in that case doubled the path
            # (e.g. ".../LULC" + "/Users/.../outputdata/output.tif"). Only
            # a relative filename typed directly into the field needs the
            # directory prefix.
            if os.path.isabs(outputText):
                self.__OutputFile = outputText
            else:
                self.__OutputFile = self.__currentDirectory + outputText
            self.ui.lePredictedFile_AccuracyAssesment.setText(self.__OutputFile);
            self.enable_ValidateDataPreparation()

    @pyqtSlot()
    def on_leT0Year_DataPreparationInputSectionDataInput_editingFinished(self):
        if(self.ui.leT0Year_DataPreparationInputSectionDataInput.isModified()):
            self.ui.leT0Year_DataPreparationInputSectionDataInput.setModified(False)
        if(not len(self.ui.leT0Year_DataPreparationInputSectionDataInput.text())) == 0:
            T0Year=int(str(self.ui.leT0Year_DataPreparationInputSectionDataInput.text().strip()))
            self.__T0Year=T0Year
            self.enable_ValidateDataPreparation()

    @pyqtSlot()
    def on_leT1Year_DataPreparationInputSectionDataInput_editingFinished(self):
        if(self.ui.leT1Year_DataPreparationInputSectionDataInput.isModified()):
            self.ui.leT1Year_DataPreparationInputSectionDataInput.setModified(False)
        if(not len(self.ui.leT1Year_DataPreparationInputSectionDataInput.text())) == 0:
            T1Year=int(str(self.ui.leT1Year_DataPreparationInputSectionDataInput.text().strip()))
            self.__T1Year=T1Year
            self.enable_ValidateDataPreparation()

    def enable_ValidateDataPreparation(self):
        """Gate on the same 5 fields the old enable_NextDataPreaparation()
        used to gate Next on directly. Next itself now only
        enables after a successful Validate click, in
        on_pbValidate_DataPreparation_clicked below."""
        if(not len(self.ui.leT0File_DataPreparationInputSectionDataInput.text()) == 0 and
               not len(self.ui.leT1File_DataPreparationInputSectionDataInput.text()) == 0 and
               not len(self.ui.leOutputFile_DataPreparationOutputSection.text()) == 0 and
               not len(self.ui.leT0Year_DataPreparationInputSectionDataInput.text()) == 0 and
               not len(self.ui.leT1Year_DataPreparationInputSectionDataInput.text()) == 0# and self.__T1Year != 0 and self.__T0Year != 0 and self.__T1Year>self.__T0Year
               ):
            self.ui.pbValidate_DataPreparation.setEnabled(True)
        else:
            self.ui.pbValidate_DataPreparation.setEnabled(False)
            self.ui.pbNextDataPreparation.setEnabled(False)

    @pyqtSlot()
    def on_pbValidate_DataPreparation_clicked(self):
        """Run LULC.accuracy.check_dataset (R: isDataSetCorrect) against the
        T0/T1 rasters before letting the user proceed past Data Preparation.
        Drivers aren't chosen until the later Driver Selection
        tabs, so they're intentionally not included here.
        AreaOfInterest/Mask are included when populated,
        but only if they're actual rasters: both fields also accept a
        shapefile elsewhere in this tab (masking._is_vector's convention),
        and a vector path isn't a comparable raster grid.
        """
        extra_layers = {}
        for name, le in (
            ("AreaOfInterest", self.ui.leAreaOfInterest_DataPreparationInputSectionDataInput),
            ("Mask", self.ui.leMask_DataPreparationInputSectionDataInput),
        ):
            path = le.text().strip()
            if path and not path.lower().endswith((".shp", ".gpkg", ".geojson", ".json")):
                extra_layers[name] = path

        report = self.controller.check_dataset(
            {}, {}, self.__T0File, self.__T1File, extra_layers=extra_layers or None,
        )

        if report.ok:
            self.ui.leT1Year_DataPreparationInputSectionDataInput.setEnabled(False)
            self.ui.leT0Year_DataPreparationInputSectionDataInput.setEnabled(False)
            self.ui.leOutputFile_DataPreparationOutputSection.setEnabled(False)
            self.ui.leT0File_DataPreparationInputSectionDataInput.setEnabled(False)
            self.ui.leT1File_DataPreparationInputSectionDataInput.setEnabled(False)
            self.ui.pbNextDataPreparation.setEnabled(True)
            QMessageBox.information(self, "Dataset Validation", "No errors found in the input datasets.")
        else:
            self.ui.pbNextDataPreparation.setEnabled(False)
            QMessageBox.warning(self, "Dataset Validation", "\n".join(report.issues))

    def preparecbInSteps_DemandAllocationSpatialContext(self):
        diffyear=self.__T1Year-self.__T0Year
        for i in list(range(1,diffyear,1)):
            self.ui.cbInSteps_DemandAllocationSpatialContext.addItem(str(i+1))
            
    def breakCoeffDetails(self, class_summary_text):
        """Parse one class's entry from LULC.modeling.get_model_summary's
        text format ("Class: X  Model: Y\\nPositive samples: N\\n(name)\\t(value)[\\t(pvalue)\\t(star)]\\n...")
        into [[driver_name, value, star], ...], intercept first.

        Replaces the original R-summary-text parser (R's summary.glm/
        summary.lm/randomForest-importance console output has a completely
        different shape — "Coefficients:"/"---"/"AIC:"/"IncNodePurity"
        markers) since get_model_summary now returns a simple, stable
        tab-separated format we control on both ends.
        Significance stars come from an auxiliary statsmodels Wald-test fit
        (logistic/nnet only, modeling._fit_logistic_pvalues) — the 4-field
        form is only emitted when that fit converged; other model types
        (and a non-converging statsmodels fit) fall back to the 2-field
        form, so ``star`` is "" here.
        """
        coeffs = {}
        for line in class_summary_text.splitlines():
            if "\t" not in line:
                continue
            parts = line.split("\t")
            name = parts[0].strip()
            value = parts[1].strip()
            star = parts[3].strip() if len(parts) >= 4 else ""
            coeffs[name] = (value, star)

        layer = [["(Intercept)", *coeffs.get("(Intercept)", ("NA", ""))]]
        for name in self.__DriverDictionaryT1.keys():
            layer.append([name, *coeffs.get(name, ("NA", ""))])
        return layer

    def checkStar(self, list1):
        # scikit-learn doesn't provide R's significance stars (***/**/*/.)
        # without an extra statsmodels fit; degrades to "no stars".
        if len(list1) > 2 and list1[-1] in ("***", "**", "*", "."):
            return list1[-1]
        return ""

    def createLULCVsDriverCoefficientMatrix(self):
        # self.modelSummary is {class_name: summary_text}, in the same
        # order as self.__className (see on_pbViewModelStatistics_...).
        summaries = list(self.modelSummary.values())
        self.__confidenceinterval = []
        for i in list(range(0,self.__noOfClasses,1)):
            details = self.breakCoeffDetails(summaries[i]) if i < len(summaries) else []
            coeffAndIntervalForAllDrivers=[]
            for j in list(range(0,self.noOfDrivers+1,1)):#Including Intercepts there are one extra looping
                coeffAndInterval=[]
                detailsDriver=details[j]
                if(len(detailsDriver)):
                    coeffAndInterval.append(detailsDriver[0]) #Name of Driver at index 0
                    coeffAndInterval.append(detailsDriver[1]) #Coeff of Driver at index 1
                    if(detailsDriver[-1]=="***" or detailsDriver[-1]=="**" or detailsDriver[-1]=="*" or detailsDriver[-1]=="."):#significant code of Driver at index -1 if present otherwise
                        _11s=detailsDriver[-1]
                    else:
                        _11s=" "
                    coeffAndInterval.append(_11s)
                    coeffAndIntervalForAllDrivers.append(coeffAndInterval)
            self.__confidenceinterval.append(coeffAndIntervalForAllDrivers)

    @pyqtSlot(bool)
    def on_twMigrationOrder_ModelAnalysis_toggled(self,checked):
        self.buildtwMigrationOrder_ModelAnalysis(self.ui.twMigrationOrder_ModelAnalysis)
        if(checked):
            self.ui.twMigrationOrder_ModelAnalysis.setEnabled(False)
            
    @pyqtSlot(bool)
    def on_rbUserDefined_ModelAnalysisMigrationOrder_toggled(self,checked):
        self.buildtwMigrationOrder_ModelAnalysis(self.ui.twMigrationOrder_ModelAnalysis)
        if(checked):
            self.__conversionOrder='UD'
            self.ui.twMigrationOrder_ModelAnalysis.setEnabled(True)
            for i in list(range(0,self.__noOfClasses,1)):
                for j in list(range(0,self.__noOfClasses,1)):
                    num=str(j+1)
                self.ui.twMigrationOrder_ModelAnalysis.item(i,j+1).setText(num)

    @pyqtSlot(QTableWidgetItem)
    def on_twMigrationOrder_ModelAnalysis_itemChanged(self,item):
        if item.column() == 0:  # class-name sync
            self._sync_class_name(item.row(), item.text())
            return
        if(item.isSelected()):
            row=self.ui.twMigrationOrder_ModelAnalysis.currentRow()
            col=self.ui.twMigrationOrder_ModelAnalysis.currentColumn()
            if(col>=1):#First Column is Class Names
                strvalue=str(item.text())
                if(strvalue.isdigit()):
                    newvalue=int(strvalue)
                    for j in list(range(0,self.__noOfClasses,1)):
                        flag=1
                        for k in list(range(0,self.__noOfClasses,1)):
                            checkitem=int(str(self.ui.twMigrationOrder_ModelAnalysis.item(row,k+1).text()))
                            if((j+1)==checkitem):
                                flag=0
                                break
                        if(flag!=0):
                            oldvalue=str(j+1)
                            break

                    if(0<newvalue<=int(str(self.__noOfClasses))):
                        for i in list(range(0,self.__noOfClasses,1)):
                            if((i+1)!=col):
                                checkitem=int(str(self.ui.twMigrationOrder_ModelAnalysis.item(row,i+1).text()))
                                if(checkitem==newvalue):
                                    newcol=i+1
                                    self.ui.twMigrationOrder_ModelAnalysis.item(row,newcol).setText(oldvalue)
                    else:
                        print ("please enter value within range:1-",self.__noOfClasses)
                        self.ui.twMigrationOrder_ModelAnalysis.item(row,col).setText(oldvalue)
                else:
                    print ("Wrong Input")
                    item.setText('0')



######
    def buildtwMigrationOrder_ModelAnalysis(self,twMigrationOrder):
        twMigrationOrder.setRowCount(self.__noOfClasses)
        twMigrationOrder.setColumnCount(self.__noOfClasses+1)
        #Setting Individual tabelitems
        row=twMigrationOrder.rowCount()
        col=twMigrationOrder.columnCount()
        twMigrationOrder.blockSignals(True)
        for i in list(range(0, row, 1)):
            for j in list((range(0, col, 1))):
                if(twMigrationOrder.item(i,j)!=0): #0 is return if item is not set
                    item = QTableWidgetItem()
                    item.setText(str(j))
                    if(col==1):
                        item.flags()^QtCore.Qt.ItemIsEditable
                    twMigrationOrder.setItem(i, j, item)

        #Setting Table Header — col 0 pulls from self.__className (the
        # class-name sync source of truth) and stays editable so
        # renaming here also propagates.
        stringlist2 = list()#QtCore.QStringList()
        stringlist2.append("Class")
        for i in list(range(0, row, 1)):
            classname=self.__className[i]
            stringlist2.append(classname)
            twMigrationOrder.item(i,0).setText(classname)
            item = twMigrationOrder.item(i,0)
            item.setFlags(QtCore.Qt.ItemIsSelectable|QtCore.Qt.ItemIsEnabled|QtCore.Qt.ItemIsEditable)

        twMigrationOrder.setHorizontalHeaderLabels(stringlist2)
        twMigrationOrder.blockSignals(False)

    def buildtwPolicies_DemandAllocation(self,twPolicies):
        #twPolicies.setRowCount(self.__noOfClasses)
        twPolicies.setRowCount((self.__noOfClasses)+1)
        twPolicies.setColumnCount(4)
        #Setting Individual tabelitems
        row=twPolicies.rowCount()
        col=twPolicies.columnCount()
        twPolicies.blockSignals(True)
        for i in list(range(0, row, 1)):
            for j in list(range(0, col, 1)):
                item = QTableWidgetItem()
                # Class column stays editable for actual class rows (class-name
                # sync); everything else, and the trailing "Total"
                # row, keep their original read-only behavior.
                if j == 0 and i < self.__noOfClasses:
                    item.setText(self.__className[i])
                else:
                    item.setFlags(item.flags()^QtCore.Qt.ItemIsEditable)
                twPolicies.setItem(i, j, item)
        #Setting Table Header
        stringlist2 = list()#QtCore.QStringList()
        stringlist2.append('Class')
        stringlist2.append('Allocation')
        stringlist2.append('Demand')
        stringlist2.append('Class Inertia')
        twPolicies.setHorizontalHeaderLabels(stringlist2)
        twPolicies.item(row-1,0).setText('Total')
        twPolicies.blockSignals(False)

    def buildtwDetailed_AccuracyAssesment(self,twClasswiseAccuracy,forConfusionmatrix):
        if(forConfusionmatrix):
            twClasswiseAccuracy.setColumnCount(len(self.__transitionMatrix))
            twClasswiseAccuracy.setRowCount(len(self.__transitionMatrix))
        else:
            twClasswiseAccuracy.setColumnCount(1)
            twClasswiseAccuracy.setRowCount(len(self.__transitionMatrix)-1)
          
        #Setting Individual tabelitems
        row=twClasswiseAccuracy.rowCount()
        col=twClasswiseAccuracy.columnCount()
        for i in list(range(0, row, 1)):
            for j in list(range(0, col, 1)):
                item = QTableWidgetItem()
                item.setFlags(item.flags()^QtCore.Qt.ItemIsEditable)
                twClasswiseAccuracy.setItem(i, j, item)
        #Setting Table Header
        stringlistv = []
        for i in list(range(0,row-1,1)):
            if(self.ui.twViewModelCoefficint_ModelAnalysis.item(i,0)):
                classname=self.ui.twViewModelCoefficint_ModelAnalysis.item(i,0).text()
            else:
                classname="class"+str(i+1)
            stringlistv.append(classname)
        if(forConfusionmatrix):
            stringlistv.append("Total")
            stringlisth=stringlistv
        else:
            stringlisth=["Accuracy"]
        twClasswiseAccuracy.setHorizontalHeaderLabels(stringlisth)   
        twClasswiseAccuracy.setVerticalHeaderLabels(stringlistv)

    def _mapClassStyle(self, plot_file):
        """(class_ids, names, colours) for the raster about to be plotted.
        Class ids come from the raster itself (not any GUI
        table — no persistent list of the original numeric codes survives
        class-name renaming), sorted; names/colours come from
        twColorTable_ViewMaps by row position, which the rest of this
        codebase already assumes lines up with sorted class-id order
        (e.g. prepareClassName()). Returns None (after showing a message)
        if the row counts don't match — plotting a file with the wrong
        number of classes for the current color table isn't meaningful.
        """
        try:
            class_ids = self.controller.get_class_codes(plot_file, na_value=self.__modelNAValue)
        except PipelineError as exc:
            QMessageBox.critical(self, "OpenLDM Error", str(exc))
            return None
        table = self.ui.twColorTable_ViewMaps
        if len(class_ids) != table.rowCount():
            QMessageBox.critical(
                self, "OpenLDM Error",
                f"{plot_file} has {len(class_ids)} classes, but the color table has "
                f"{table.rowCount()} rows — select/rebuild a matching color table first.",
            )
            return None
        names = [table.item(i, 1).text() for i in range(table.rowCount())]
        colours = [table.item(i, 2).background().color().name() for i in range(table.rowCount())]
        return class_ids, names, colours

    @pyqtSlot()
    def on_pbShow_ViewMaps_clicked(self):
        plot_file = self.getFileToPlot()
        if not plot_file:
            return
        style = self._mapClassStyle(plot_file)
        if style is None:
            return
        class_ids, names, colours = style
        try:
            png_path = os.path.join(tempfile.gettempdir(), "openldm_view_map.png")
            self.controller.generate_map(
                plot_file, png_path, class_ids, names, colours,
                title=self.ui.leTitle_ViewMaps.text(),
                legend_title=self.ui.leLegendHeading_ViewMaps.text(),
                na_value=self.__modelNAValue,
            )
        except Exception as exc:
            QMessageBox.critical(self, "OpenLDM Error", f"Could not render map: {exc}")
            return
        pixmap = QtGui.QPixmap(png_path)
        self.ui.lbCanvas_ViewMaps.setPixmap(
            pixmap.scaled(self.ui.lbCanvas_ViewMaps.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        )

    @pyqtSlot()
    def on_pbExport_ViewMaps_clicked(self):
        plot_file = self.getFileToPlot()
        if not plot_file:
            return
        style = self._mapClassStyle(plot_file)
        if style is None:
            return
        class_ids, names, colours = style
        output_file, _ = QFileDialog.getSaveFileName(
            self, "Export Map", self.__currentDirectory, "PNG Image (*.png)"
        )
        if not output_file:
            return
        if not output_file.lower().endswith(".png"):
            output_file += ".png"
        try:
            self.controller.generate_map(
                plot_file, output_file, class_ids, names, colours,
                title=self.ui.leTitle_ViewMaps.text(),
                legend_title=self.ui.leLegendHeading_ViewMaps.text(),
                na_value=self.__modelNAValue,
            )
        except Exception as exc:
            QMessageBox.critical(self, "OpenLDM Error", f"Could not export map: {exc}")
            return
        QMessageBox.information(self, "Export complete", f"Wrote {output_file}")

    def buildtwColorTable_ViewMaps(self,twColorTable_ViewMaps):
        twColorTable_ViewMaps.setRowCount(self.__noOfClasses)
        twColorTable_ViewMaps.setColumnCount(3)
        row=twColorTable_ViewMaps.rowCount()
        col=twColorTable_ViewMaps.columnCount()
        m=int(255/(row-1))
        twColorTable_ViewMaps.blockSignals(True)
        for i in list(range(0, row)):
            for j in list(range(0, col)):
                item = QTableWidgetItem()
                if j == 0:
                    item.setText(str(self.__className[i]))
                elif j== 1 :
                    item.setText('Class-'+str(i+1))
                else:
                    item.setFlags(item.flags()^QtCore.Qt.ItemIsEditable)
                    item.setBackground(QBrush(QColor(255-m*i,255-m*i,255-m*i)))
                    self.SelectColour_pushed();
                twColorTable_ViewMaps.setItem(i, j, item)
        twColorTable_ViewMaps.blockSignals(False)
        # Moved here from a since-removed on_twColorTable_ViewMaps_itemChanged
        # tail (it referenced the bare name `twColorTable_ViewMaps` instead
        # of `self.ui.twColorTable_ViewMaps`, so it NameError'd on every
        # edit — headers only need setting once, at construction, anyway.
        # Labels corrected to match what's actually in each column (col 0
        # is the synced class name, not a DN/pixel value).
        twColorTable_ViewMaps.setHorizontalHeaderLabels(["Class Name", "Legend Text", "Colour"])
        twColorTable_ViewMaps.resizeColumnsToContents()

    # --- Class-name sync across the 5 tables that each carry their own
    # "Class" name column: twViewModelCoefficint_ModelAnalysis,
    # twSelectModelTypeAndDrivers_DriverSelectionT1,
    # twMigrationOrder_ModelAnalysis, twPolicies_DemandAllocation, and
    # twColorTable_ViewMaps col 0. Each independently defaulted to its own
    # placeholder text with no link between them, so renaming a class in
    # one table silently didn't affect the others — self.__className is
    # the single source of truth; edit any of the five and the other four
    # (and self.__className itself, which prepareClassName()/Execute
    # ultimately read) update to match.
    __CLASS_NAME_TABLES = (
        "twViewModelCoefficint_ModelAnalysis",
        "twSelectModelTypeAndDrivers_DriverSelectionT1",
        "twMigrationOrder_ModelAnalysis",
        "twPolicies_DemandAllocation",
        "twColorTable_ViewMaps",
    )

    def _sync_class_name(self, row, new_name):
        if row >= len(self.__className):
            return
        self.__className[row] = new_name
        for table_name in self.__CLASS_NAME_TABLES:
            table = getattr(self.ui, table_name)
            if row >= table.rowCount():
                continue
            item = table.item(row, 0)
            if item is not None and item.text() != new_name:
                table.blockSignals(True)
                item.setText(new_name)
                table.blockSignals(False)
        # twMigrationOrder_ModelAnalysis also names each class again as a
        # column header (col 0 is "migrate FROM this class", the header of
        # column row+1 is "migrate TO this class") — not covered by the
        # col-0-only loop above.
        header = self.ui.twMigrationOrder_ModelAnalysis.horizontalHeaderItem(row + 1)
        if header is not None and header.text() != new_name:
            header.setText(new_name)

    @pyqtSlot(QTableWidgetItem)
    def on_twViewModelCoefficint_ModelAnalysis_itemChanged(self, item):
        if item.column() == 0:
            self._sync_class_name(item.row(), item.text())

    @pyqtSlot(QTableWidgetItem)
    def on_twSelectModelTypeAndDrivers_DriverSelectionT1_itemChanged(self, item):
        if item.column() == 0:
            self._sync_class_name(item.row(), item.text())

    @pyqtSlot(QTableWidgetItem)
    def on_twColorTable_ViewMaps_itemChanged(self, item):
        if item.column() == 0:
            self._sync_class_name(item.row(), item.text())

    def on_twColorTable_ViewMaps_cellDoubleClicked(self,row,col):
        if(col==2):
            color = QColorDialog.getColor()
            item=self.ui.twColorTable_ViewMaps.item(row,col)
            item.setBackground(QBrush(color))
        self.ui.twColorTable_ViewMaps.resizeColumnsToContents()

    def SelectColour_pushed(self):
        self.btn = QPushButton('Dialog', self)
        self.btn.clicked.connect(self.showDialog)

    def getFileToPlot(self):
        ClassifiedRasterFile=str(self.ui.cbFile_ViewMaps.currentText())
        if(ClassifiedRasterFile=="" or not QFileInfo(ClassifiedRasterFile).isReadable()):
            reply=QMessageBox.question(self, "File doesn't Exist?","Do you want to add a file?",QMessageBox.Yes | QMessageBox.No, QMessageBox.No);
            if(reply==QMessageBox.Yes):
                ClassifiedRasterFile,_ = QFileDialog.getOpenFileName(self, "Open File",self.__currentDirectory,"Raster (*.tif *.img)")
                lineEdit=QLineEdit()
                lineEdit.setText(ClassifiedRasterFile)
                self.ui.cbFile_ViewMaps.setLineEdit(lineEdit)
                self.ui.cbFile_ViewMaps.addItem(ClassifiedRasterFile)
                ClassifiedRasterFile = str(ClassifiedRasterFile)
            else:
                ClassifiedRasterFile = ""
        return ClassifiedRasterFile

    # prepareDataForPlot/getLegendClassName/getLegendClassValue/prepareTempPng
    # (R genrateMap-based cartographic rendering) removed — deferred.
    # getFileToPlot/buildtwColorTable_ViewMaps above stay in place for the
    # next slice to build on (class/color table population doesn't depend
    # on R).

    def showDialog(self):
        col = QtGui.QColorDialog.getColor()
        if col.isValid():
            self.frm.setStyleSheet("QWidget { background-color: %s }"% col.name())

    def prepareExecutionEnv(self):
        installed_dir = self.get_main_dir()
        installed_dir=installed_dir.replace(os.sep,"/")
        self.__projectDirectory = installed_dir
        self.ui.leProjectDirectory_DataPreparationInputSectionProjectSection.setText(str(self.__projectDirectory))
        self.__installedDir=installed_dir

    # prepareR/releaseR (R sourcing + CheckInstallPackage) and getColorPallet
    # (fed R's genrateMap) removed — no R backend to prepare/release anymore,
    # and the color table isn't consumed by anything yet that cartography
    # is deferred.

    def closeEvent(self, event):
        quit_msg = "Are you sure you want to exit the program?"
        reply = QtWidgets.QMessageBox.question(self, 'OpenLDM Message',
                                                   quit_msg, QtWidgets.QMessageBox.Yes, QtWidgets.QMessageBox.No)

        if reply == QtWidgets.QMessageBox.Yes:
            self.statusMessage.exit(0)
            log_bridge.detach(self._log_handler)
            event.accept()
        else:
            event.ignore()

    def _result_available(self, ok):
        frame = self.page().mainFrame()

    def main_is_frozen(self):
        return (hasattr(sys, "frozen") or # new py2exe
                    hasattr(sys, "importers") # old py2exe
                    #or imp.is_frozen("__main__")
                    ) # tools/freeze

    def get_main_dir(self):
        if self.main_is_frozen():
            return os.path.dirname(os.path.abspath(sys.executable))
        # sys.argv[0] (or the launcher's own last arg) is whatever
        # *launched* this Python process -- correct for `python
        # OpenLDM.py`, but meaningless when embedded as a QGIS plugin,
        # where sys.argv[0] reflects QGIS's own binary, not this file's
        # location. This module's own __file__ (gui/main_window.py) is a
        # reliable anchor in every case: its parent's parent is the same
        # "src" directory (helpdoc/, gui/, LULC/ live as siblings there)
        # this method is meant to return, standalone or embedded alike.
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    def filldebugInput(self):
        self.ui.leT0File_DataPreparationInputSectionDataInput.setText(str("../examples/LULC/1985.tif"))   
        self.__T0File=str(self.ui.leT0File_DataPreparationInputSectionDataInput.text())
        
        self.ui.leT0Year_DataPreparationInputSectionDataInput.setText(str("1985"))
        self.__T0Year = int(str(self.ui.leT0Year_DataPreparationInputSectionDataInput.text()))
        
        self.ui.leT1File_DataPreparationInputSectionDataInput.setText(str("../examples/LULC/1995.tif"))
        self.__T1File=str(self.ui.leT1File_DataPreparationInputSectionDataInput.text())
        
        self.ui.leT1Year_DataPreparationInputSectionDataInput.setText(str("1995"))
        self.__T1Year = int(str(self.ui.leT1Year_DataPreparationInputSectionDataInput.text()))
        
        self.ui.leOutputFile_DataPreparationOutputSection.setText(str("../examples/outputdata/modeloutput.tif"))
        self.__OutputFile= str(self.ui.leOutputFile_DataPreparationOutputSection.text())
        self.ui.lePredictedFile_AccuracyAssesment.setText(self.__OutputFile)
        
        
        
    def filldebugAdddriver(self):
        self.filldebugSelectDriver(1,"../examples/Drivers/commonDrivers/elevation.img")
        self.filldebugSelectDriver(2,"../examples/Drivers/drivers_85/dist_stream.img")
        self.filldebugSelectDriver(3,"../examples/Drivers/drivers_85/Dist_urban.img")
        self.filldebugSelectDriver(4,"../examples/Drivers/drivers_85/road_final.img")
        
    def filldebugSelectDriver(self,row,filename):
        self.on_pbAddDriver_DriverSelectionT0SelectDrivers_clicked()
        disp=self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(row-1, 1)
        disp.setText(filename)
        disp=self.ui.twSelectDrivers_DriverSelectionT0.cellWidget(row-1, 0)
        (dirName, OnlyFilename) = os.path.split(filename.strip())
        disp.setText(OnlyFilename.replace(".","_"))

        
    def filldebug(self):
        self.filldebugInput()
        self.on_pbNextDataPreparation_clicked()

        self.filldebugAdddriver()
        self.fillModelType()
        self.on_pbViewModelStatistics_DriverSelectionT0DoModelFitting_clicked()
        self.on_pbNext_DriverSelectionT0_clicked()

        self.fillClassNames()
        self.on_pbNext_ModelAnalysis_clicked()
        
    
    def fillClassNames(self):
        twViewModelCoefficint=self.ui.twViewModelCoefficint_ModelAnalysis 
        row=twViewModelCoefficint.rowCount()
        j=0
        classnames=['BU','AG','DF','FL','GL','MF','PL','SL','WB']
        for i in list(range(0, row, 1)):
            twViewModelCoefficint.item(i,j).setText(classnames[i])

    def fillModelType(self):
        self.ui.rbLogisticRegression_DriverSelectionT0DoModelFitting.setChecked(True)
        
        
        
    def printParameter(self):
        print("model.type-[[ "+str(self.__modeltype)+ " ]] ")
        print("T0File-[[ "+self.__T0File+ " ]] ")
        print("T1File-[[ "+self.__T1File+ " ]] ")
        print("T0Year-[[ "+str(self.__T0Year)+ " ]] ")
        print("T1Year-[[ "+str(self.__T1Year)+ " ]] ")        
        print("T1drivers-[[ "+str(self.__DriverDictionaryT1)+ " ]] ")
        print("T2drivers-[[ "+str(self.__DriverDictionaryT2)+ " ]] ")
        print("withNAvalue-[[ "+str(self.__modelNAValue)+ " ]] ")
        print("modelformula-[[ "+str(self.__modelformula)+ " ]] ")
        print("outputfile-[[ "+self.__OutputFile+ " ]] ")
        print("modelNAValue-[[ "+str(self.__modelNAValue)+ " ]] ")
        print("MASKFile-[[ "+str(self.__MASKFile)+ " ]] ")
        print("AOIFile-[[ "+str(self.__AOIFile)+ " ]] ")
        print("suitabilityFileDirectory-[[ "+str(self.__suitabilityFileDirectory)+ " ]] ")


LULCMainWindow = MyForm  # public name used by OpenLDM.py's run_gui() and future callers


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    myapp = MyForm()
    myapp.show()
    myapp.setWindowIcon(QtGui.QIcon(":/images/images/icon.png"))
    r=app.exec_()
    print(r)
    sys.exit(r)
