"""Bridges the ``LULC`` package's existing ``logging`` calls to Qt signals.

The processing layer (``LULC.config.logger``, ``"OpenLDM"``) already calls
``logger.info(...)`` at every meaningful pipeline stage — data loading,
model fitting per class, suitability prediction, each allocation phase,
writing outputs (see ``LULCAlgorithms.run_pipeline`` and the modules it
calls). Attaching a handler here gives the GUI real, live progress text for
free, replacing the original GUI's fake elapsed-time-only status bar
(``StatusBarThread``) and the R backend's non-functional ``setRStatus``/
``getRStatus`` — without threading a progress callback through
every function signature in the processing layer.
"""

from __future__ import annotations

import logging

from PyQt5 import QtCore

from LULC.config import logger as lulc_logger


class QtLogHandler(QtCore.QObject, logging.Handler):
    """A ``logging.Handler`` that emits a Qt signal per record.

    Must inherit ``QObject`` (for the signal) alongside ``logging.Handler``;
    the handler itself is not thread-affine — ``emit`` is called from
    whichever thread is running the logged code (typically a
    :class:`gui.workers.BackgroundTaskWorker`), and Qt's queued-connection
    signal delivery marshals it back to the GUI thread automatically as long
    as this object was constructed on the GUI thread.
    """

    message_logged = QtCore.pyqtSignal(str, int)  # (formatted message, levelno)

    def __init__(self, level: int = logging.INFO) -> None:
        QtCore.QObject.__init__(self)
        logging.Handler.__init__(self, level=level)
        self.setFormatter(logging.Formatter("%(message)s"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.message_logged.emit(self.format(record), record.levelno)
        except Exception:
            self.handleError(record)


def attach(level: int = logging.INFO) -> QtLogHandler:
    """Attach a fresh :class:`QtLogHandler` to the ``LULC`` logger and
    return it (caller owns it — connect to ``message_logged`` and keep a
    reference for the GUI's lifetime; call :func:`detach` on shutdown)."""
    handler = QtLogHandler(level=level)
    lulc_logger.addHandler(handler)
    return handler


def detach(handler: QtLogHandler) -> None:
    lulc_logger.removeHandler(handler)
