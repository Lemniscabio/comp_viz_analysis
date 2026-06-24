"""QThread for frame grabbing with frame-dropping design."""

from __future__ import annotations

import threading
import time
import logging

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal

logger = logging.getLogger(__name__)


class CaptureThread(QThread):
    """Dedicated thread for frame grabbing.

    Always overwrites the shared frame buffer with the latest frame.
    Never queues frames — latency matters more than completeness.
    """

    frame_ready = pyqtSignal()

    def __init__(self, device_index: int, resolution: tuple[int, int] = (1280, 720)) -> None:
        super().__init__()
        self._device_index = device_index
        self._resolution = resolution
        self._frame: np.ndarray | None = None
        self._lock = threading.Lock()
        self._running = False
        self._reconnect_interval = 2.0  # seconds between reconnect attempts

    def run(self) -> None:
        self._running = True
        cap: cv2.VideoCapture | None = None

        while self._running:
            if cap is None or not cap.isOpened():
                cap = self._try_open()
                if cap is None:
                    logger.warning("Camera not available, retrying in %.1fs", self._reconnect_interval)
                    time.sleep(self._reconnect_interval)
                    continue

            ret, frame = cap.read()
            if ret and frame is not None:
                with self._lock:
                    self._frame = frame
                self.frame_ready.emit()
            else:
                logger.warning("Camera read failed — attempting reconnect")
                cap.release()
                cap = None

        if cap is not None:
            cap.release()

    def _try_open(self) -> cv2.VideoCapture | None:
        cap = cv2.VideoCapture(self._device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        # DirectShow (and virtual webcams like Camo) commonly return black frames
        # for the first several frames after opening. Discard them before the main
        # loop starts serving frames to the UI.
        for _ in range(10):
            cap.grab()
        logger.info("Capture thread opened device %d", self._device_index)
        return cap

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self) -> None:
        self._running = False
        self.wait(5000)  # ms timeout
