"""Generic background-task worker.

The original GUI ran every R call synchronously on the Qt main thread,
freezing the UI during model fitting/allocation.
:class:`BackgroundTaskWorker` runs an arbitrary callable on a QThread and
reports back via signals, so the two genuinely slow operations ("View Model
Statistics" and "Execute") no longer block the GUI.
"""

from __future__ import annotations

from typing import Any, Callable

from PyQt5 import QtCore


class BackgroundTaskWorker(QtCore.QThread):
    """Runs ``func(*args, **kwargs)`` on a background thread.

    Emits exactly one of ``finished`` or ``failed`` when done. Progress
    updates arrive separately via :mod:`gui.log_bridge`, not through this
    class — keeping this worker generic (it doesn't need to know anything
    about the LULC pipeline's internal stages).
    """

    finished_ok = QtCore.pyqtSignal(object)  # result
    failed = QtCore.pyqtSignal(Exception)

    def __init__(self, func: Callable[..., Any], *args: Any, parent=None, **kwargs: Any) -> None:
        super().__init__(parent)
        self._func = func
        self._args = args
        self._kwargs = kwargs

    def run(self) -> None:
        try:
            result = self._func(*self._args, **self._kwargs)
        except Exception as exc:  # noqa: BLE001 - deliberately broad, forwarded to the GUI thread
            self.failed.emit(exc)
        else:
            self.finished_ok.emit(result)
