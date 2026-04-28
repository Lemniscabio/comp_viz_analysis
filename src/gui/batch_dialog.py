"""Modal batch-analysis dialog: pick videos, output dir, scale option, run."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt6.QtWidgets import (
    QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel, QListWidget,
    QProgressBar, QPushButton, QVBoxLayout,
)

from src.core.mixing_time import MixingTimeParams
from src.gui.batch_worker import BatchWorker

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


class BatchDialog(QDialog):
    """Modal dialog for batch-analyzing many videos with the current config."""

    def __init__(
        self,
        config: dict,
        roi,
        mask: Optional[np.ndarray],
        params: MixingTimeParams,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Batch Analyze Videos")
        self.setMinimumSize(560, 480)

        self._config = config
        self._roi = roi
        self._mask = mask
        self._params = params
        self._videos: List[Path] = []
        self._output_dir: Optional[Path] = None
        self._worker: Optional[BatchWorker] = None

        layout = QVBoxLayout(self)

        layout.addWidget(QLabel("Videos:"))
        self._list = QListWidget()
        layout.addWidget(self._list)

        row = QHBoxLayout()
        btn_add_files = QPushButton("Add Files…")
        btn_add_files.clicked.connect(self._on_add_files)
        btn_add_folder = QPushButton("Add Folder…")
        btn_add_folder.clicked.connect(self._on_add_folder)
        btn_clear = QPushButton("Clear")
        btn_clear.clicked.connect(self._on_clear)
        row.addWidget(btn_add_files)
        row.addWidget(btn_add_folder)
        row.addWidget(btn_clear)
        layout.addLayout(row)

        row2 = QHBoxLayout()
        btn_out = QPushButton("Choose Output Dir…")
        btn_out.clicked.connect(self._on_choose_out)
        self._lbl_out = QLabel("(no output dir)")
        row2.addWidget(btn_out)
        row2.addWidget(self._lbl_out, 1)
        layout.addLayout(row2)

        self._chk_scale = QCheckBox("Scale ROI/mask if video size differs")
        self._chk_scale.setChecked(True)
        layout.addWidget(self._chk_scale)

        self._chk_per_video = QCheckBox("Export per-video metrics CSV")
        self._chk_per_video.setChecked(True)
        layout.addWidget(self._chk_per_video)

        self._progress = QProgressBar()
        self._progress.setVisible(False)
        layout.addWidget(self._progress)
        self._lbl_status = QLabel("")
        layout.addWidget(self._lbl_status)

        row3 = QHBoxLayout()
        self._btn_run = QPushButton("Run")
        self._btn_run.setStyleSheet(
            "QPushButton:enabled { background-color: #2d7d46; color: white; "
            "font-weight: bold; padding: 6px 16px; }"
        )
        self._btn_run.clicked.connect(self._on_run)
        self._btn_close = QPushButton("Close")
        self._btn_close.clicked.connect(self.reject)
        row3.addStretch()
        row3.addWidget(self._btn_run)
        row3.addWidget(self._btn_close)
        layout.addLayout(row3)

    # --- pickers ---
    def _on_add_files(self) -> None:
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Videos", "",
            "Videos (*.mp4 *.avi *.mov *.mkv *.m4v);;All (*)",
        )
        for f in files:
            p = Path(f)
            self._videos.append(p)
            self._list.addItem(str(p))

    def _on_add_folder(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not d:
            return
        for p in sorted(Path(d).iterdir()):
            if p.suffix.lower() in VIDEO_EXTS:
                self._videos.append(p)
                self._list.addItem(str(p))

    def _on_clear(self) -> None:
        self._videos.clear()
        self._list.clear()

    def _on_choose_out(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Output Directory")
        if d:
            self._output_dir = Path(d)
            self._lbl_out.setText(d)

    # --- run ---
    def _on_run(self) -> None:
        if not self._videos:
            self._lbl_status.setText("No videos selected")
            return
        if self._output_dir is None:
            self._lbl_status.setText("Choose an output directory")
            return
        self._btn_run.setEnabled(False)
        self._progress.setVisible(True)
        self._progress.setMaximum(len(self._videos))
        self._progress.setValue(0)
        self._worker = BatchWorker(
            videos=self._videos,
            output_dir=self._output_dir,
            config=self._config,
            roi=self._roi,
            mask=self._mask,
            params=self._params,
            scale_roi=self._chk_scale.isChecked(),
            export_per_video_csv=self._chk_per_video.isChecked(),
        )
        self._worker.progress.connect(self._on_progress)
        self._worker.video_done.connect(self._on_video_done)
        self._worker.finished_all.connect(self._on_finished_all)
        self._worker.error_occurred.connect(
            lambda e: self._lbl_status.setText(f"Error: {e}")
        )
        self._worker.start()

    def _on_progress(self, i: int, total: int, name: str) -> None:
        self._progress.setValue(i)
        self._lbl_status.setText(f"[{i}/{total}] {name}")

    def _on_video_done(self, name: str, status: str) -> None:
        self._lbl_status.setText(f"{name}: {status}")

    def _on_finished_all(self, summary_csv: str) -> None:
        self._btn_run.setEnabled(True)
        self._lbl_status.setText(f"Done. Summary: {summary_csv}")

    def closeEvent(self, event) -> None:
        if self._worker and self._worker.isRunning():
            self._worker.stop()
            self._worker.wait(3000)
        super().closeEvent(event)
