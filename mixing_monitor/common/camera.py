"""Camera device enumeration and frame capture."""

import logging
import cv2
import numpy as np

from .constants import DEFAULT_RESOLUTION, CAMERA_PROBE_RANGE

logger = logging.getLogger(__name__)


def _enumerate_with_pygrabber() -> list[dict]:
    """Return device list using pygrabber (Windows DirectShow)."""
    from pygrabber.dshow_graph import FilterGraph  # type: ignore

    graph = FilterGraph()
    names: list[str] = graph.get_input_devices()
    return [{"index": i, "name": name} for i, name in enumerate(names)]


def _enumerate_by_probing() -> list[dict]:
    """Return device list by probing OpenCV indices 0-9."""
    devices: list[dict] = []
    for idx in CAMERA_PROBE_RANGE:
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            devices.append({"index": idx, "name": f"Camera {idx}"})
            cap.release()
    return devices


class CameraSource:
    """Wraps a single camera device for frame capture."""

    def __init__(self) -> None:
        self._cap: cv2.VideoCapture | None = None
        self._actual_resolution: tuple[int, int] = (0, 0)
        self._actual_fps: float = 0.0

    # ------------------------------------------------------------------
    # Static helpers
    # ------------------------------------------------------------------

    @staticmethod
    def list_devices() -> list[dict]:
        """Return available video devices as list of {index, name} dicts.

        Tries pygrabber first (gives friendly device names on Windows).
        Falls back to probing indices 0-9 via OpenCV if pygrabber is missing.
        """
        try:
            devices = _enumerate_with_pygrabber()
            if devices:
                return devices
        except Exception as exc:
            logger.debug("pygrabber enumeration failed (%s), falling back to probe", exc)
        return _enumerate_by_probing()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, device_index: int, resolution: tuple[int, int] = DEFAULT_RESOLUTION) -> bool:
        """Open device at the given index.

        Requests the specified resolution; accepts whatever the device gives.
        Returns True on success.
        """
        self.release()
        cap = cv2.VideoCapture(device_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            logger.warning("Failed to open camera index %d", device_index)
            return False

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        if (actual_w, actual_h) != resolution:
            logger.warning(
                "Camera %d does not support %dx%d; using %dx%d",
                device_index, resolution[0], resolution[1], actual_w, actual_h,
            )

        # Discard the first few frames — DirectShow and virtual webcams (Camo)
        # often return black frames immediately after opening.
        for _ in range(10):
            cap.grab()

        self._cap = cap
        self._actual_resolution = (actual_w, actual_h)
        self._actual_fps = actual_fps if actual_fps > 0 else 30.0
        logger.info(
            "Opened camera %d at %dx%d @ %.1ffps",
            device_index, actual_w, actual_h, self._actual_fps,
        )
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read one frame from the device. Returns (success, BGR frame)."""
        if self._cap is None or not self._cap.isOpened():
            return False, None
        ret, frame = self._cap.read()
        if not ret:
            logger.debug("Camera read failed (ret=False)")
        return ret, frame if ret else None

    def release(self) -> None:
        """Release the device."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_opened(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def actual_resolution(self) -> tuple[int, int]:
        return self._actual_resolution

    @property
    def actual_fps(self) -> float:
        return self._actual_fps
