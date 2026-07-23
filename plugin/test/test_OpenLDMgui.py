# coding=utf-8
"""Smoke tests for the OpenLDM plugin's run() -- launching the real
desktop GUI (gui.main_window.LULCMainWindow) as a window inside QGIS,
rather than the old empty Plugin Builder dialog stub.

.. note:: This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

"""

__author__ = 'jha.ashutosh@gmail.com'
__date__ = '2026-07-21'
__copyright__ = 'Copyright 2026, Indian Institute of Remote Sensing'

import sys
import unittest
from pathlib import Path
from unittest import mock

# `plugin/OpenLDMgui.py` uses package-relative imports (`from . import
# dependency_check`), same as it will once QGIS loads the deployed plugin
# folder as a package -- so it must be imported as `plugin.OpenLDMgui`
# here too, not as a bare top-level module (unlike the old dialog test,
# which imported its target as a bare module and so never actually
# exercised the real package-relative import wiring).
REPO_ROOT = str(Path(__file__).resolve().parents[2])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from qgis.PyQt import QtWidgets  # noqa: E402
from qgis.PyQt import QtWebEngineWidgets  # noqa: E402  (must import before QApplication, same Qt quirk as tests/conftest.py)

from utilities import get_qgis_app  # noqa: E402

QGIS_APP, CANVAS, IFACE, PARENT = get_qgis_app()


class _StubWebEngineView(QtWidgets.QWidget):
    """Same stub tests/conftest.py uses for the main suite: QWebEngineView
    hangs constructing its Chromium renderer subprocess in this sandboxed
    environment. Applied to PyQt5 specifically since gui.main_window
    imports PyQt5 directly (not qgis.PyQt) -- it's the standalone app's
    own dependency, unrelated to whichever Qt binding QGIS itself uses."""

    def setHtml(self, *args, **kwargs):
        pass


class OpenLDMPluginTest(unittest.TestCase):
    """Test run() launches the real window, safely, when embedded."""

    def setUp(self):
        from PyQt5 import QtWebEngineWidgets as PyQt5WebEngine
        PyQt5WebEngine.QWebEngineView = _StubWebEngineView

        from plugin.OpenLDMgui import OpenLDM
        self.plugin = OpenLDM(IFACE)

        # No dialogs block a headless test run. LULCMainWindow's own
        # construction can hit QMessageBox.information via loadHelpFile()
        # if the help file doesn't resolve under this test's cwd (same
        # dialog gui.main_window's own test suite guards against in
        # tests/test_gui_smoke.py's main_window fixture) -- both patched
        # via qgis.PyQt since QMessageBox is the same class object
        # whichever import path (qgis.PyQt or plain PyQt5) reaches it.
        self._question_patch = mock.patch(
            "qgis.PyQt.QtWidgets.QMessageBox.question",
            return_value=QtWidgets.QMessageBox.Yes,
        )
        self._question_patch.start()
        self._information_patch = mock.patch(
            "qgis.PyQt.QtWidgets.QMessageBox.information",
            return_value=QtWidgets.QMessageBox.Ok,
        )
        self._information_patch.start()

        # Dependencies are whatever's importable in the interpreter
        # actually running this test -- not what a real end user's QGIS
        # Python has. The install flow itself (QThread + pip) is tested
        # separately and directly in test_dependency_check.py; here we
        # just want run() to proceed straight to launching the window.
        self._deps_patch = mock.patch(
            "plugin.dependency_check.missing_packages", return_value=[]
        )
        self._deps_patch.start()

    def tearDown(self):
        self.plugin.unload()
        self._deps_patch.stop()
        self._information_patch.stop()
        self._question_patch.stop()

    def test_run_launches_the_real_desktop_window(self):
        self.plugin.run()
        self.assertIsNotNone(self.plugin._window)
        from gui.main_window import LULCMainWindow
        self.assertIsInstance(self.plugin._window, LULCMainWindow)
        self.assertTrue(self.plugin._window.isVisible())

    def test_run_twice_reuses_the_same_window(self):
        self.plugin.run()
        first = self.plugin._window
        self.plugin.run()
        self.assertIs(self.plugin._window, first)

    def test_exit_action_does_not_quit_the_host_application(self):
        """Regression test for the embedding-safety fix: File > Exit
        inside the embedded window must close only that window, not call
        QApplication.quit() and take QGIS down with it."""
        self.plugin.run()
        window = self.plugin._window

        with mock.patch("PyQt5.QtWidgets.QApplication.quit") as quit_mock:
            window.on_actionExit_triggered()
            quit_mock.assert_not_called()


if __name__ == "__main__":
    suite = unittest.makeSuite(OpenLDMPluginTest)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
