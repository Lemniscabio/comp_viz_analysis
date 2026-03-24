"""Video file reader wrapping OpenCV VideoCapture."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterator, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("kineticolor")


class VideoReader:
    """Reads frames from a video file with frame skipping support."""

    def __init__(
        self,
        path: Path | str,
        frame_skip: int = 1,
        fps_override: Optional[float] = None,
    ) -> None:
        self._path = Path(path)
        self._frame_skip = frame_skip
        self._cap = cv2.VideoCapture(str(self._path))

        if not self._cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {self._path}")

        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._native_fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps_override if fps_override is not None else self._native_fps
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._current_frame = 0

        logger.info(
            f"Opened video: {self._path.name} "
            f"({self._frame_count} frames, {self._fps:.1f} fps, "
            f"{self._width}x{self._height})"
        )

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def current_frame(self) -> int:
        return self._current_frame

    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read the next frame, respecting frame_skip."""
        while True:
            ret, frame = self._cap.read()
            if not ret:
                return False, None
            self._current_frame += 1
            if (self._current_frame - 1) % self._frame_skip == 0:
                return True, frame

    def seek(self, frame_number: int) -> None:
        """Seek to a specific frame number."""
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        self._current_frame = frame_number

    def get_frame(self, frame_number: int) -> Optional[np.ndarray]:
        """Read a specific frame by number without advancing the reader position."""
        saved_pos = self._current_frame
        self.seek(frame_number)
        ret, frame = self._cap.read()
        self.seek(saved_pos)
        return frame if ret else None

    def timestamp(self, frame_number: int) -> float:
        """Convert frame number to timestamp in seconds."""
        if self._fps <= 0:
            return 0.0
        return frame_number / self._fps

    def __iter__(self) -> Iterator[Tuple[int, np.ndarray]]:
        """Iterate over frames, yielding (frame_number, frame)."""
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            yield self._current_frame - 1, frame

    def release(self) -> None:
        """Release the video capture resource."""
        if hasattr(self, "_cap") and self._cap is not None:
            self._cap.release()

    def __del__(self) -> None:
        self.release()
