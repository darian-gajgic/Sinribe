"""QThread wrapper around pipeline.Runner.

The Runner calls its callbacks from this worker thread; every one of them is turned into a Qt
signal, which is the only thread-safe way to reach the GUI. No widget is ever touched here.
"""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal

from ..pipeline import Cancelled, JobSpec, Runner


class JobWorker(QObject):
    progress = Signal(float, str, str)   # fraction 0..1, phase, detail
    log = Signal(str)
    finished = Signal(dict)
    failed = Signal(str, str)            # message, kind
    cancelled = Signal()

    def __init__(self, spec: JobSpec):
        super().__init__()
        self.spec = spec
        self.runner = Runner(
            spec,
            on_progress=lambda f, p, d: self.progress.emit(f, p, d),
            on_log=lambda m: self.log.emit(m),
        )

    def run(self) -> None:
        try:
            result = self.runner.run()
        except Cancelled:
            self.cancelled.emit()
        except Exception as e:  # noqa: BLE001 - surface everything in the UI, never crash
            kind = getattr(e, "kind", "runtime")
            self.failed.emit(f"{type(e).__name__}: {e}", kind)
        else:
            self.finished.emit(result)

    def cancel(self) -> None:
        self.runner.cancel()


class ProbeWorker(QObject):
    """Looks up a URL's title/duration without downloading.

    Runs on its own thread because it is a network round-trip, and a few seconds of frozen
    window while yt-dlp talks to YouTube reads as a crash.
    """
    done = Signal(object)     # fetch.MediaMeta
    failed = Signal(str)

    def __init__(self, url: str, cookies: str | None):
        super().__init__()
        self.url = url
        self.cookies = cookies

    def run(self) -> None:
        from .. import fetch
        try:
            self.done.emit(fetch.probe(self.url, self.cookies))
        except Exception as e:  # noqa: BLE001
            self.failed.emit(str(e))


def start_probe(url: str, cookies: str | None) -> tuple[QThread, ProbeWorker]:
    thread = QThread()
    worker = ProbeWorker(url, cookies)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    worker.done.connect(thread.quit)
    worker.failed.connect(thread.quit)
    thread.start()
    return thread, worker


def start_job(spec: JobSpec) -> tuple[QThread, JobWorker]:
    """Create and start a thread running `spec`. Caller keeps references to both."""
    thread = QThread()
    worker = JobWorker(spec)
    worker.moveToThread(thread)
    thread.started.connect(worker.run)
    for sig in (worker.finished, worker.failed, worker.cancelled):
        sig.connect(thread.quit)
    thread.start()
    return thread, worker
