"""Background thread workers used by the SAGA segmentation GUI."""

import subprocess
from PyQt5.QtCore import QThread, pyqtSignal


class Worker(QThread):
    """Run a callable in a background thread; emit (result, error_str) when done."""

    finished = pyqtSignal(object, str)

    def __init__(self, fn, *args, **kwargs):
        super().__init__()
        self._fn = fn
        self._args = args
        self._kwargs = kwargs

    def run(self):
        try:
            result = self._fn(*self._args, **self._kwargs)
            self.finished.emit(result, "")
        except Exception:
            import traceback
            self.finished.emit(None, traceback.format_exc())


class SubprocessWorker(QThread):
    """Run a subprocess and emit stdout/stderr lines as they arrive."""

    line_ready = pyqtSignal(str)
    finished_proc = pyqtSignal(int)   # return code

    def __init__(self, cmd: list, cwd: str):
        super().__init__()
        self._cmd = cmd
        self._cwd = cwd
        self._proc = None

    def run(self):
        self._proc = subprocess.Popen(
            self._cmd, cwd=self._cwd,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
            encoding="utf-8", errors="replace",
        )
        for line in self._proc.stdout:
            self.line_ready.emit(line.rstrip())
        self._proc.wait()
        self.finished_proc.emit(self._proc.returncode)

    def terminate_process(self):
        if self._proc and self._proc.poll() is None:
            self._proc.terminate()