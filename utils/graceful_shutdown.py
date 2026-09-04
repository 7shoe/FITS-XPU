"""Cooperative process shutdown for Aurora XPU entry points.

Signal handlers only record the request.  The training/evaluation loops raise
``GracefulShutdown`` between batches, after any in-flight XPU operation has
completed, so normal Python and Level Zero teardown can run.
"""

from __future__ import annotations

import signal
import threading


class GracefulShutdown(Exception):
    """Raised at a safe batch boundary after SIGINT or SIGTERM."""

    def __init__(self, signum: int):
        self.signum = int(signum)
        super().__init__("graceful shutdown requested by signal {}".format(signum))


_lock = threading.Lock()
_requested_signal = None


def _record_signal(signum, _frame):
    global _requested_signal
    with _lock:
        if _requested_signal is None:
            _requested_signal = int(signum)
            print(
                "GRACEFUL_XPU_SHUTDOWN requested: signal={}; "
                "waiting for a safe batch boundary".format(signum),
                flush=True,
            )


def install_signal_handlers():
    """Install non-raising SIGINT/SIGTERM handlers and return this module."""

    signal.signal(signal.SIGINT, _record_signal)
    signal.signal(signal.SIGTERM, _record_signal)


def requested_signal():
    """Return the first requested signal number, or ``None``."""

    with _lock:
        return _requested_signal


def raise_if_requested():
    """Raise at a caller-selected safe point when shutdown was requested."""

    signum = requested_signal()
    if signum is not None:
        raise GracefulShutdown(signum)
