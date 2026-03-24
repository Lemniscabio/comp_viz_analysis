"""Background worker thread for running analysis engine."""
from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

from src.core.analysis_engine import AnalysisEngine
from src.core.video_reader import VideoReader
from src.utils.config_loader import load_config

logger = logging.getLogger("kineticolor")


class BrightnessWarningHandler(logging.Handler):
    """Captures brightness warnings and forwards them via a callback."""

    def __init__(self, callback):
        super().__init__(level=logging.WARNING)
        self._callback = callback

    def emit(self, record):
        if "brightness" in record.getMessage().lower():
            self._callback(record.getMessage())


class AnalysisWorker(QThread):
    """Runs AnalysisEngine in a background thread."""

    frame_ready = pyqtSignal(int, np.ndarray, np.ndarray, dict)
    progress = pyqtSignal(int, int)
    analysis_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    brightness_warning = pyqtSignal(str)

    def __init__(
        self,
        config: Dict[str, Any],
        video_path: Optional[str] = None,
        camera_index: Optional[int] = None,
        roi: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[np.ndarray] = None,
        reference_frame_num: int = 0,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self._config = config
        self._video_path = video_path
        self._camera_index = camera_index
        self._roi = roi
        self._mask = mask
        self._reference_frame_num = reference_frame_num
        self._running = True
        self._engine: Optional[AnalysisEngine] = None

    @property
    def engine(self) -> Optional[AnalysisEngine]:
        return self._engine

    def stop(self) -> None:
        self._running = False

    def run(self) -> None:
        try:
            warn_handler = BrightnessWarningHandler(
                lambda msg: self.brightness_warning.emit(msg)
            )
            logging.getLogger("kineticolor").addHandler(warn_handler)

            if self._video_path:
                reader = VideoReader(
                    path=self._video_path,
                    frame_skip=self._config["frame_skip"],
                    fps_override=self._config.get("video_fps_override"),
                )
                total = reader.frame_count // self._config["frame_skip"]
                is_camera = False
            elif self._camera_index is not None:
                reader = VideoReader(
                    path=self._camera_index,
                    frame_skip=self._config["frame_skip"],
                )
                total = 0
                is_camera = True
            else:
                self.error_occurred.emit("No video source specified")
                return

            self._engine = AnalysisEngine(self._config)

            if self._reference_frame_num > 0 and not is_camera:
                ref = reader.get_frame(self._reference_frame_num)
                if ref is not None:
                    self._engine.set_reference_frame_data(ref)

            processed = 0
            reconnect_attempts = 0
            max_reconnect = 10

            while self._running:
                ret, frame = reader.read_frame()

                if not ret:
                    if is_camera and reconnect_attempts < max_reconnect:
                        reconnect_attempts += 1
                        self.error_occurred.emit(
                            f"Camera disconnected. Reconnecting... ({reconnect_attempts}/{max_reconnect})"
                        )
                        self.msleep(2000)
                        reader = VideoReader(
                            path=self._camera_index,
                            frame_skip=self._config["frame_skip"],
                        )
                        continue
                    break

                reconnect_attempts = 0
                frame_number = reader.current_frame - 1
                timestamp = reader.timestamp(frame_number)

                result = self._engine.process_frame(
                    frame, frame_number, timestamp,
                    roi=self._roi, mask=self._mask,
                )

                pixel_de = result.get("pixel_delta_e", np.zeros((1, 1)))
                self.frame_ready.emit(frame_number, frame, pixel_de, result)
                processed += 1
                self.progress.emit(processed, total)

            reader.release()
            logging.getLogger("kineticolor").removeHandler(warn_handler)
            self.analysis_finished.emit()

        except Exception as e:
            logger.exception("Analysis worker error")
            self.error_occurred.emit(str(e))
