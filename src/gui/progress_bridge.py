"""Bridges LULCAlgorithms.run_pipeline's ``on_progress`` callback to a Qt signal.

Mirrors log_bridge.py's shape exactly: a small QObject relay whose method is
called synchronously from whichever thread is running the pipeline
(typically a gui.workers.BackgroundTaskWorker), emitting a Qt signal that
Qt's queued-connection delivery marshals back to the GUI thread
automatically. This is a real, quantitative percentage (LULCAlgorithms.py's
own stage boundaries), complementing log_bridge's live log-text forwarding
rather than replacing it — the processing layer's internals
(modeling/allocation/transition) still don't need to know about either.
"""

from __future__ import annotations

from PyQt5 import QtCore


class QtProgressRelay(QtCore.QObject):
    """``report(percent, label)`` emits ``progress_changed`` — pass
    ``self.report`` as ``on_progress`` to run_pipeline/generate_predicted_map."""

    progress_changed = QtCore.pyqtSignal(int, str)

    def report(self, percent: int, label: str) -> None:
        self.progress_changed.emit(percent, label)
