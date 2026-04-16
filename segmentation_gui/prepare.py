"""
PrepareDialog — reusable dialog for running SAGA pipeline preparation steps.

Shows a parameter form, a live subprocess log, and Run / Close buttons.
Each preparation step (masks, scale, contrastive) instantiates this dialog
with its own parameters and command builder.
"""

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout,
    QTextEdit, QPushButton,
)
from PyQt5.QtGui import QFont

from .workers import SubprocessWorker


class PrepareDialog(QDialog):
    """
    Generic dialog for a single pipeline preparation step.

    Usage::

        dlg = PrepareDialog("Step Title", parent)
        dlg.add_param("Label:", some_widget)
        dlg.set_run(command_fn, cwd)   # command_fn() -> list[str]
        dlg.exec_()

    The dialog streams subprocess stdout/stderr into a QTextEdit log area.
    """

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(680, 500)
        self._cmd_fn = None
        self._cwd = ""
        self._worker: SubprocessWorker | None = None

        root = QVBoxLayout(self)
        root.setSpacing(8)

        # Parameter form
        self._form = QFormLayout()
        self._form.setRowWrapPolicy(QFormLayout.WrapLongRows)
        self._form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        root.addLayout(self._form)

        # Live log output
        self._log = QTextEdit()
        self._log.setReadOnly(True)
        self._log.setFont(QFont("Courier New", 8))
        self._log.setMinimumHeight(240)
        root.addWidget(self._log)

        # Buttons
        btn_row = QHBoxLayout()
        self._run_btn = QPushButton("Run")
        self._run_btn.setDefault(True)
        self._run_btn.clicked.connect(self._on_run)
        btn_row.addWidget(self._run_btn)
        self._close_btn = QPushButton("Close")
        self._close_btn.clicked.connect(self.close)
        btn_row.addWidget(self._close_btn)
        root.addLayout(btn_row)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_param(self, label: str, widget) -> None:
        """Add a labelled row to the parameter form."""
        self._form.addRow(label, widget)

    def set_run(self, command_fn, cwd: str) -> None:
        """
        Set the command builder and working directory.

        ``command_fn`` is called with no arguments when Run is clicked;
        it must return a list[str] suitable for subprocess.Popen.
        """
        self._cmd_fn = command_fn
        self._cwd = cwd

    # ------------------------------------------------------------------
    # Slots
    # ------------------------------------------------------------------

    def _on_run(self):
        if self._worker and self._worker.isRunning():
            return
        try:
            cmd = self._cmd_fn()
        except Exception as e:
            self._log.append(f"[Error] Could not build command: {e}")
            return

        self._log.clear()
        self._log.append("$ " + " ".join(str(c) for c in cmd))
        self._log.append("")
        self._run_btn.setEnabled(False)

        self._worker = SubprocessWorker(cmd, self._cwd)
        self._worker.line_ready.connect(self._append_line)
        self._worker.finished_proc.connect(self._on_finished)
        self._worker.start()

    def _append_line(self, line: str):
        self._log.append(line)
        sb = self._log.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_finished(self, rc: int):
        self._run_btn.setEnabled(True)
        if rc == 0:
            self._log.append("\n[Done] Completed successfully.")
        else:
            self._log.append(f"\n[Error] Process exited with code {rc}.")

    def closeEvent(self, event):
        if self._worker and self._worker.isRunning():
            self._worker.terminate_process()
            self._worker.wait(2000)
        super().closeEvent(event)