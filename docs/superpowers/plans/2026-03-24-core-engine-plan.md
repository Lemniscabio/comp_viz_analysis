# Core Engine Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the headless core computation engine for Kineticolor — video ingestion, frame processing, 6 mixing metrics, grid analysis, and data export — fully testable with synthetic images.

**Architecture:** Layered core engine with no GUI dependency. VideoReader ingests frames, FrameProcessor handles ROI/mask/color conversion, AnalysisEngine orchestrates 6 metrics per frame storing results as time series, DataExporter writes CSV/XLSX. All metrics follow a BaseMetric interface. GLCM is computed once and shared by 3 metrics.

**Tech Stack:** Python 3.11+, OpenCV, NumPy, SciPy, scikit-image, PyYAML, openpyxl, pytest

---

## File Structure

```
requirements.txt
config/default_config.yaml
src/__init__.py
src/main.py
src/utils/__init__.py
src/utils/logger.py
src/utils/color_convert.py
src/utils/config_loader.py
src/core/__init__.py
src/core/video_reader.py
src/core/frame_processor.py
src/core/grid_analyzer.py
src/core/metrics/__init__.py
src/core/metrics/base_metric.py
src/core/metrics/delta_e.py
src/core/metrics/contact.py
src/core/metrics/glcm.py
src/core/metrics/contrast.py
src/core/metrics/homogeneity.py
src/core/metrics/energy.py
src/core/metrics/variance.py
src/core/analysis_engine.py
src/core/export.py
tests/__init__.py
tests/fixtures/README.md
tests/test_config_loader.py
tests/test_color_convert.py
tests/test_frame_processor.py
tests/test_grid_analyzer.py
tests/test_delta_e.py
tests/test_contact.py
tests/test_glcm.py
tests/test_contrast.py
tests/test_homogeneity.py
tests/test_energy.py
tests/test_variance.py
tests/test_analysis_engine.py
tests/test_export.py
tests/test_integration.py
```

---

### Task 1: Project scaffolding and dependencies

**Files:**
- Create: `requirements.txt`
- Create: `config/default_config.yaml`
- Create: `src/__init__.py`, `src/core/__init__.py`, `src/core/metrics/__init__.py`, `src/utils/__init__.py`
- Create: `tests/__init__.py`, `tests/fixtures/README.md`

- [ ] **Step 1: Create requirements.txt**

```
numpy>=1.24
opencv-python>=4.8
scikit-image>=0.21
scipy>=1.11
pyyaml>=6.0
openpyxl>=3.1
pytest>=7.4
```

- [ ] **Step 2: Create config/default_config.yaml**

```yaml
# Frame processing
frame_skip: 1                    # Analyze every Nth frame (1 = every frame)
glcm_frame_skip: 1              # GLCM-specific frame skip (can be higher than frame_skip)

# Grid
grid_rows: 5                    # Number of rows in spatial grid
grid_cols: 5                    # Number of columns in spatial grid

# GLCM
glcm_gray_levels: 16            # Number of quantized gray levels (fewer = faster)
glcm_offset: [1, 1]             # Pixel pair offset [dx, dy]

# Contact
contact_threshold: 128          # Grayscale threshold for binary contact analysis (0-255)

# Video / Camera
camera_index: 0                 # Default camera device index for live feed
video_fps_override: null        # Override video FPS if metadata is wrong

# Brightness warning
brightness_change_threshold: 0.2  # Warn if brightness changes by more than 20%

# Export
export_format: "csv"            # csv or xlsx
```

- [ ] **Step 3: Create all `__init__.py` files and fixtures README**

All `__init__.py` files are empty. `tests/fixtures/README.md` contains: "Test fixtures: synthetic images and short video clips for unit tests."

- [ ] **Step 4: Install dependencies and verify**

Run: `pip install -r requirements.txt`
Expected: All packages install successfully.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt config/ src/__init__.py src/core/__init__.py src/core/metrics/__init__.py src/utils/__init__.py tests/__init__.py tests/fixtures/
git commit -m "scaffold: project structure, dependencies, and config"
```

---

### Task 2: Logger utility

**Files:**
- Create: `src/utils/logger.py`

- [ ] **Step 1: Implement setup_logger**

```python
"""Centralized logging setup with console and rotating file handlers."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from datetime import datetime


def setup_logger(
    name: str = "kineticolor",
    log_dir: str = "logs",
    console_level: int = logging.INFO,
    file_level: int = logging.DEBUG,
) -> logging.Logger:
    """Configure and return a logger with console and rotating file handlers.

    Args:
        name: Logger name.
        log_dir: Directory for log files.
        console_level: Minimum level for console output.
        file_level: Minimum level for file output.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    file_handler = RotatingFileHandler(
        log_path / f"kineticolor_{timestamp}.log",
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=5,
    )
    file_handler.setLevel(file_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.utils.logger import setup_logger; log = setup_logger(); log.info('Logger works')"`
Expected: Console prints the log line, `logs/` directory created with log file.

- [ ] **Step 3: Commit**

```bash
git add src/utils/logger.py
git commit -m "feat: add centralized logger with rotating file handler"
```

---

### Task 3: Config loader

**Files:**
- Create: `src/utils/config_loader.py`
- Create: `tests/test_config_loader.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for config loader."""

import pytest
from src.utils.config_loader import load_config, DEFAULT_CONFIG


class TestLoadConfig:
    def test_load_defaults(self):
        """Loading with no file returns all default values."""
        config = load_config()
        assert config["frame_skip"] == 1
        assert config["glcm_frame_skip"] == 1
        assert config["grid_rows"] == 5
        assert config["grid_cols"] == 5
        assert config["glcm_gray_levels"] == 16
        assert config["glcm_offset"] == [1, 1]
        assert config["contact_threshold"] == 128
        assert config["camera_index"] == 0
        assert config["video_fps_override"] is None
        assert config["export_format"] == "csv"
        assert config["brightness_change_threshold"] == 0.2

    def test_load_from_yaml(self, tmp_path):
        """Loading from a YAML file overrides defaults."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("frame_skip: 3\ngrid_rows: 10\n")
        config = load_config(yaml_file)
        assert config["frame_skip"] == 3
        assert config["grid_rows"] == 10
        assert config["grid_cols"] == 5  # unchanged default

    def test_invalid_frame_skip(self, tmp_path):
        """frame_skip < 1 raises ValueError."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("frame_skip: 0\n")
        with pytest.raises(ValueError, match="frame_skip"):
            load_config(yaml_file)

    def test_invalid_contact_threshold(self, tmp_path):
        """contact_threshold outside 0-255 raises ValueError."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("contact_threshold: 300\n")
        with pytest.raises(ValueError, match="contact_threshold"):
            load_config(yaml_file)

    def test_invalid_export_format(self, tmp_path):
        """export_format not csv or xlsx raises ValueError."""
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text("export_format: json\n")
        with pytest.raises(ValueError, match="export_format"):
            load_config(yaml_file)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_config_loader.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement config_loader.py**

```python
"""YAML config loader with defaults and validation."""

from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "frame_skip": 1,
    "glcm_frame_skip": 1,
    "grid_rows": 5,
    "grid_cols": 5,
    "glcm_gray_levels": 16,
    "glcm_offset": [1, 1],
    "contact_threshold": 128,
    "camera_index": 0,
    "video_fps_override": None,
    "brightness_change_threshold": 0.2,
    "export_format": "csv",
}


def _validate(config: dict[str, Any]) -> None:
    """Validate config values, raising ValueError on invalid entries."""
    if config["frame_skip"] < 1:
        raise ValueError(f"frame_skip must be >= 1, got {config['frame_skip']}")
    if config["glcm_frame_skip"] < 1:
        raise ValueError(f"glcm_frame_skip must be >= 1, got {config['glcm_frame_skip']}")
    if config["grid_rows"] < 1:
        raise ValueError(f"grid_rows must be >= 1, got {config['grid_rows']}")
    if config["grid_cols"] < 1:
        raise ValueError(f"grid_cols must be >= 1, got {config['grid_cols']}")
    if config["glcm_gray_levels"] < 2:
        raise ValueError(f"glcm_gray_levels must be >= 2, got {config['glcm_gray_levels']}")
    if not (0 <= config["contact_threshold"] <= 255):
        raise ValueError(f"contact_threshold must be 0-255, got {config['contact_threshold']}")
    if config["export_format"] not in ("csv", "xlsx"):
        raise ValueError(f"export_format must be 'csv' or 'xlsx', got '{config['export_format']}'")


def load_config(path: Path | str | None = None) -> dict[str, Any]:
    """Load config from YAML file, falling back to defaults.

    Args:
        path: Path to YAML config file. If None, uses defaults only.

    Returns:
        Validated config dict.
    """
    config = DEFAULT_CONFIG.copy()

    if path is not None:
        path = Path(path)
        with open(path) as f:
            user_config = yaml.safe_load(f)
        if user_config:
            config.update(user_config)

    _validate(config)
    return config
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_config_loader.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/config_loader.py tests/test_config_loader.py
git commit -m "feat: add YAML config loader with validation"
```

---

### Task 4: Color conversion utilities

**Files:**
- Create: `src/utils/color_convert.py`
- Create: `tests/test_color_convert.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for color conversion utilities."""

import numpy as np
from src.utils.color_convert import rgb_to_lab, lab_to_rgb


class TestRgbToLab:
    def test_black(self):
        """Pure black RGB -> L*=0, a*~0, b*~0."""
        img = np.zeros((10, 10, 3), dtype=np.uint8)
        lab = rgb_to_lab(img)
        assert lab.shape == (10, 10, 3)
        assert np.allclose(lab[:, :, 0], 0, atol=0.5)

    def test_white(self):
        """Pure white RGB -> L*=100, a*~0, b*~0."""
        img = np.full((10, 10, 3), 255, dtype=np.uint8)
        lab = rgb_to_lab(img)
        assert np.allclose(lab[:, :, 0], 100, atol=0.5)

    def test_output_dtype(self):
        """Output should be float64."""
        img = np.zeros((5, 5, 3), dtype=np.uint8)
        lab = rgb_to_lab(img)
        assert lab.dtype == np.float64


class TestLabToRgb:
    def test_roundtrip(self):
        """RGB -> LAB -> RGB should be close to original."""
        img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        lab = rgb_to_lab(img)
        recovered = lab_to_rgb(lab)
        assert recovered.dtype == np.uint8
        assert np.allclose(recovered, img, atol=2)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_color_convert.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement color_convert.py**

```python
"""RGB <-> CIE-L*a*b* color space conversion helpers."""

import numpy as np
from skimage.color import rgb2lab, lab2rgb


def rgb_to_lab(image: np.ndarray) -> np.ndarray:
    """Convert an RGB uint8 image to CIE-L*a*b* float64.

    Args:
        image: (H, W, 3) uint8 array in RGB color space.

    Returns:
        (H, W, 3) float64 array in L*a*b* color space.
    """
    return rgb2lab(image)


def lab_to_rgb(image: np.ndarray) -> np.ndarray:
    """Convert a CIE-L*a*b* float64 image to RGB uint8.

    Args:
        image: (H, W, 3) float64 array in L*a*b* color space.

    Returns:
        (H, W, 3) uint8 array in RGB color space.
    """
    rgb_float = lab2rgb(image)
    return np.clip(rgb_float * 255, 0, 255).astype(np.uint8)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_color_convert.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/utils/color_convert.py tests/test_color_convert.py
git commit -m "feat: add RGB <-> CIE-L*a*b* conversion utilities"
```

---

### Task 5: Frame processor

**Files:**
- Create: `src/core/frame_processor.py`
- Create: `tests/test_frame_processor.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for frame processor."""

import logging
import numpy as np
import pytest
from src.core.frame_processor import FrameProcessor


class TestCropToRoi:
    def test_crop_dimensions(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        fp = FrameProcessor()
        cropped = fp.crop_to_roi(frame, (10, 20, 50, 30))  # x, y, w, h
        assert cropped.shape == (30, 50, 3)

    def test_crop_content(self):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        frame[10:20, 10:20] = 255
        fp = FrameProcessor()
        cropped = fp.crop_to_roi(frame, (10, 10, 10, 10))
        assert np.all(cropped == 255)

    def test_none_roi_returns_full_frame(self):
        frame = np.zeros((50, 60, 3), dtype=np.uint8)
        fp = FrameProcessor()
        cropped = fp.crop_to_roi(frame, None)
        assert cropped.shape == frame.shape


class TestApplyMask:
    def test_mask_zeros_pixels(self):
        frame = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[5:, :] = 0  # mask out bottom half
        fp = FrameProcessor()
        result = fp.apply_mask(frame, mask)
        assert np.all(result[5:, :] == 0)
        assert np.all(result[:5, :] == 128)

    def test_none_mask_returns_copy(self):
        frame = np.full((10, 10, 3), 100, dtype=np.uint8)
        fp = FrameProcessor()
        result = fp.apply_mask(frame, None)
        assert np.array_equal(result, frame)


class TestBrightnessCheck:
    def test_no_warning_on_stable_brightness(self, caplog):
        fp = FrameProcessor(brightness_change_threshold=0.2)
        frame1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 105, dtype=np.uint8)
        fp.check_brightness(frame1, None)
        with caplog.at_level(logging.WARNING):
            fp.check_brightness(frame2, None)
        assert "brightness" not in caplog.text.lower()

    def test_warning_on_drastic_change(self, caplog):
        fp = FrameProcessor(brightness_change_threshold=0.2)
        frame1 = np.full((10, 10, 3), 100, dtype=np.uint8)
        frame2 = np.full((10, 10, 3), 200, dtype=np.uint8)
        fp.check_brightness(frame1, None)
        with caplog.at_level(logging.WARNING):
            fp.check_brightness(frame2, None)
        assert "brightness" in caplog.text.lower()

    def test_mask_excluded_from_brightness(self):
        fp = FrameProcessor(brightness_change_threshold=0.2)
        frame = np.full((10, 10, 3), 200, dtype=np.uint8)
        frame[5:, :] = 0
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[5:, :] = 0
        fp.check_brightness(frame, mask)
        # Brightness should reflect only unmasked region (value 200)
        assert abs(fp._prev_brightness - 200.0) < 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_frame_processor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement frame_processor.py**

```python
"""Frame processing: ROI extraction, mask application, color space conversion."""

import logging

import cv2
import numpy as np

from src.utils.color_convert import rgb_to_lab

logger = logging.getLogger("kineticolor")


class FrameProcessor:
    """Handles ROI cropping, exclusion mask application, and color conversion."""

    def __init__(self, brightness_change_threshold: float = 0.2) -> None:
        self._brightness_change_threshold = brightness_change_threshold
        self._prev_brightness: float | None = None

    def crop_to_roi(
        self, frame: np.ndarray, roi: tuple[int, int, int, int] | None
    ) -> np.ndarray:
        """Crop frame to region of interest.

        Args:
            frame: (H, W, 3) image array.
            roi: (x, y, width, height) or None for full frame.

        Returns:
            Cropped image array.
        """
        if roi is None:
            return frame.copy()
        x, y, w, h = roi
        return frame[y : y + h, x : x + w].copy()

    def apply_mask(
        self, frame: np.ndarray, mask: np.ndarray | None
    ) -> np.ndarray:
        """Apply exclusion mask to frame. Masked-out pixels become 0.

        Args:
            frame: (H, W, 3) image array.
            mask: (H, W) uint8 array where 1=keep, 0=exclude. None means keep all.

        Returns:
            Masked image array.
        """
        if mask is None:
            return frame.copy()
        return frame * mask[:, :, np.newaxis]

    def to_lab(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to CIE-L*a*b*."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb_to_lab(rgb)

    def to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale."""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def check_brightness(
        self, frame: np.ndarray, mask: np.ndarray | None
    ) -> None:
        """Check for drastic brightness changes between consecutive frames.

        Logs a WARNING if average brightness changes by more than the threshold.

        Args:
            frame: (H, W, 3) or (H, W) image array.
            mask: (H, W) uint8 array where 1=keep, 0=exclude. None means all.
        """
        if frame.ndim == 3:
            gray = np.mean(frame, axis=2)
        else:
            gray = frame.astype(np.float64)

        if mask is not None:
            valid = mask > 0
            if not np.any(valid):
                return
            avg_brightness = float(np.mean(gray[valid]))
        else:
            avg_brightness = float(np.mean(gray))

        if self._prev_brightness is not None and self._prev_brightness > 0:
            change = abs(avg_brightness - self._prev_brightness) / self._prev_brightness
            if change > self._brightness_change_threshold:
                logger.warning(
                    f"Brightness changed by {change:.1%} "
                    f"({self._prev_brightness:.1f} -> {avg_brightness:.1f}). "
                    f"Check lighting consistency."
                )

        self._prev_brightness = avg_brightness
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_frame_processor.py -v`
Expected: All 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/frame_processor.py tests/test_frame_processor.py
git commit -m "feat: add frame processor with ROI, mask, and brightness check"
```

---

### Task 6: Video reader

**Files:**
- Create: `src/core/video_reader.py`

- [ ] **Step 1: Implement VideoReader**

```python
"""Video file reader wrapping OpenCV VideoCapture."""

import logging
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

logger = logging.getLogger("kineticolor")


class VideoReader:
    """Reads frames from a video file with frame skipping support."""

    def __init__(
        self,
        path: Path | str,
        frame_skip: int = 1,
        fps_override: float | None = None,
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

    def read_frame(self) -> tuple[bool, np.ndarray | None]:
        """Read the next frame, respecting frame_skip.

        Returns:
            (success, frame) where frame is BGR uint8 or None on failure.
        """
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

    def get_frame(self, frame_number: int) -> np.ndarray | None:
        """Read a specific frame by number without advancing the reader position.

        Args:
            frame_number: Zero-based frame index.

        Returns:
            BGR frame array, or None if the frame can't be read.
        """
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

    def __iter__(self) -> Iterator[tuple[int, np.ndarray]]:
        """Iterate over frames, yielding (frame_number, frame)."""
        while True:
            ret, frame = self.read_frame()
            if not ret:
                break
            yield self._current_frame - 1, frame

    def release(self) -> None:
        """Release the video capture resource."""
        self._cap.release()

    def __del__(self) -> None:
        self.release()
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.core.video_reader import VideoReader; print('OK')"`
Expected: Prints "OK".

- [ ] **Step 3: Commit**

```bash
git add src/core/video_reader.py
git commit -m "feat: add video reader with frame skip and fps override"
```

---

### Task 7: Grid analyzer

**Files:**
- Create: `src/core/grid_analyzer.py`
- Create: `tests/test_grid_analyzer.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for grid analyzer."""

import numpy as np
from src.core.grid_analyzer import GridAnalyzer


class TestGridAnalyzer:
    def test_cell_coordinates(self):
        """5x5 grid on a 100x100 image gives 20x20 cells."""
        ga = GridAnalyzer(rows=5, cols=5)
        cells = ga.get_cell_coords(height=100, width=100)
        assert len(cells) == 25
        # First cell
        assert cells[0] == (0, 0, 20, 20)  # (y, x, h, w)
        # Last cell
        assert cells[24] == (80, 80, 20, 20)

    def test_cell_averages_uniform(self):
        """Uniform image: all cell averages should be equal."""
        ga = GridAnalyzer(rows=5, cols=5)
        img = np.full((100, 100, 3), 128, dtype=np.uint8)
        avgs = ga.compute_cell_averages(img)
        assert avgs.shape == (25, 3)
        assert np.allclose(avgs, 128.0)

    def test_cell_averages_different_quadrants(self):
        """Image with 4 distinct quadrants should produce different averages."""
        ga = GridAnalyzer(rows=2, cols=2)
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        img[:50, :50] = [255, 0, 0]
        img[:50, 50:] = [0, 255, 0]
        img[50:, :50] = [0, 0, 255]
        img[50:, 50:] = [255, 255, 0]
        avgs = ga.compute_cell_averages(img)
        assert avgs.shape == (4, 3)
        assert np.allclose(avgs[0], [255, 0, 0])
        assert np.allclose(avgs[1], [0, 255, 0])

    def test_masked_cell_flagging(self):
        """Cells >50% masked are flagged invalid."""
        ga = GridAnalyzer(rows=2, cols=2)
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[:50, :50] = 0  # fully mask top-left cell
        valid = ga.get_valid_cells(mask, height=100, width=100)
        assert valid[0] == False
        assert valid[1] == True
        assert valid[2] == True
        assert valid[3] == True

    def test_cell_averages_with_mask(self):
        """Masked pixels excluded from cell average computation."""
        ga = GridAnalyzer(rows=1, cols=1)
        img = np.full((10, 10, 3), 100, dtype=np.uint8)
        img[5:, :] = 200
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[5:, :] = 0  # mask out the 200 region
        avgs = ga.compute_cell_averages(img, mask)
        assert np.allclose(avgs[0], 100.0)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_grid_analyzer.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement grid_analyzer.py**

```python
"""Grid analyzer: divides ROI into N×N cells for spatial analysis."""

import numpy as np


class GridAnalyzer:
    """Divides an image region into a grid of cells."""

    def __init__(self, rows: int = 5, cols: int = 5) -> None:
        self._rows = rows
        self._cols = cols

    @property
    def rows(self) -> int:
        return self._rows

    @property
    def cols(self) -> int:
        return self._cols

    def get_cell_coords(
        self, height: int, width: int
    ) -> list[tuple[int, int, int, int]]:
        """Get (y, x, h, w) coordinates for each cell.

        Returns:
            List of (y, x, h, w) tuples, row-major order.
        """
        cell_h = height // self._rows
        cell_w = width // self._cols
        cells = []
        for r in range(self._rows):
            for c in range(self._cols):
                y = r * cell_h
                x = c * cell_w
                h = cell_h if r < self._rows - 1 else height - y
                w = cell_w if c < self._cols - 1 else width - x
                cells.append((y, x, h, w))
        return cells

    def compute_cell_averages(
        self,
        image: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Compute average color per cell.

        Args:
            image: (H, W, C) image array.
            mask: (H, W) uint8 array, 1=keep, 0=exclude. None means keep all.

        Returns:
            (N_cells, C) array of per-cell average values.
        """
        h, w = image.shape[:2]
        channels = image.shape[2] if image.ndim == 3 else 1
        cells = self.get_cell_coords(h, w)
        averages = np.zeros((len(cells), channels), dtype=np.float64)

        for i, (cy, cx, ch, cw) in enumerate(cells):
            cell_data = image[cy : cy + ch, cx : cx + cw]
            if mask is not None:
                cell_mask = mask[cy : cy + ch, cx : cx + cw]
                valid = cell_mask > 0
                if not np.any(valid):
                    averages[i] = np.nan
                    continue
                if cell_data.ndim == 3:
                    averages[i] = np.mean(cell_data[valid], axis=0)
                else:
                    averages[i, 0] = np.mean(cell_data[valid])
            else:
                if cell_data.ndim == 3:
                    averages[i] = np.mean(
                        cell_data.reshape(-1, channels), axis=0
                    )
                else:
                    averages[i, 0] = np.mean(cell_data)

        return averages

    def get_valid_cells(
        self, mask: np.ndarray, height: int, width: int
    ) -> list[bool]:
        """Determine which cells are valid (>50% unmasked).

        Args:
            mask: (H, W) uint8 array, 1=keep, 0=exclude.
            height: Image height.
            width: Image width.

        Returns:
            List of booleans, one per cell.
        """
        cells = self.get_cell_coords(height, width)
        valid = []
        for cy, cx, ch, cw in cells:
            cell_mask = mask[cy : cy + ch, cx : cx + cw]
            fraction_unmasked = np.mean(cell_mask > 0)
            valid.append(fraction_unmasked > 0.5)
        return valid
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_grid_analyzer.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/grid_analyzer.py tests/test_grid_analyzer.py
git commit -m "feat: add grid analyzer with cell decomposition and masking"
```

---

### Task 8: Base metric and Delta E metric

**Files:**
- Create: `src/core/metrics/base_metric.py`
- Create: `src/core/metrics/delta_e.py`
- Create: `tests/test_delta_e.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for Delta E metric."""

import numpy as np
from src.core.metrics.delta_e import DeltaEMetric


class TestDeltaE:
    def test_identical_frames_zero(self):
        """Same frame compared to itself -> grand ΔE = 0."""
        metric = DeltaEMetric(grid_rows=5, grid_cols=5)
        lab = np.full((100, 100, 3), [50.0, 0.0, 0.0])
        result = metric.compute(lab, lab)
        assert result["grand_delta_e"] == 0.0

    def test_known_delta_e(self):
        """Two solid L*a*b* colors with known distance."""
        metric = DeltaEMetric(grid_rows=5, grid_cols=5)
        frame = np.full((10, 10, 3), [50.0, 10.0, 0.0])
        ref = np.full((10, 10, 3), [50.0, 0.0, 0.0])
        result = metric.compute(frame, ref)
        assert abs(result["grand_delta_e"] - 10.0) < 0.01

    def test_pixel_delta_e_shape(self):
        """pixel_delta_e has same H, W as input."""
        metric = DeltaEMetric(grid_rows=2, grid_cols=2)
        lab = np.random.rand(20, 30, 3) * 100
        ref = np.random.rand(20, 30, 3) * 100
        result = metric.compute(lab, ref)
        assert result["pixel_delta_e"].shape == (20, 30)

    def test_row_col_averages(self):
        """Row and column averages have correct shapes."""
        metric = DeltaEMetric(grid_rows=5, grid_cols=5)
        lab = np.random.rand(100, 100, 3) * 100
        ref = np.random.rand(100, 100, 3) * 100
        result = metric.compute(lab, ref)
        assert result["row_avg"].shape == (5,)
        assert result["col_avg"].shape == (5,)
        assert result["cell_avg"].shape == (25,)

    def test_with_mask(self):
        """Masked pixels excluded from grand ΔE."""
        metric = DeltaEMetric(grid_rows=1, grid_cols=1)
        frame = np.full((10, 10, 3), [80.0, 0.0, 0.0])
        ref = np.full((10, 10, 3), [50.0, 0.0, 0.0])
        # Mask out everything except one pixel
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[0, 0] = 1
        result = metric.compute(frame, ref, mask)
        assert abs(result["grand_delta_e"] - 30.0) < 0.01
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_delta_e.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement base_metric.py**

```python
"""Abstract base class for all mixing metrics."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class BaseMetric(ABC):
    """Base class that all metrics inherit from."""

    @abstractmethod
    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute metric values for a single frame.

        Args:
            frame: Processed frame (color space depends on metric).
            reference_frame: Reference frame (same color space).
            mask: (H, W) uint8 array, 1=keep, 0=exclude. None means keep all.

        Returns:
            Dict of named metric values.
        """
```

- [ ] **Step 4: Implement delta_e.py**

```python
"""Delta E metric: perceptually uniform color distance from reference frame."""

from typing import Any

import numpy as np

from src.core.grid_analyzer import GridAnalyzer
from src.core.metrics.base_metric import BaseMetric


class DeltaEMetric(BaseMetric):
    """Computes per-pixel Euclidean distance in CIE-L*a*b* space from reference."""

    def __init__(self, grid_rows: int = 5, grid_cols: int = 5) -> None:
        self._grid = GridAnalyzer(rows=grid_rows, cols=grid_cols)

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute Delta E between frame and reference in L*a*b* space.

        Args:
            frame: (H, W, 3) L*a*b* float64 array.
            reference_frame: (H, W, 3) L*a*b* float64 array.
            mask: (H, W) uint8 array, 1=keep, 0=exclude.

        Returns:
            Dict with grand_delta_e, pixel_delta_e, row_avg, col_avg, cell_avg.
        """
        diff = frame.astype(np.float64) - reference_frame.astype(np.float64)
        pixel_de = np.sqrt(np.sum(diff ** 2, axis=2))

        if mask is not None:
            valid = mask > 0
            if not np.any(valid):
                grand = 0.0
            else:
                grand = float(np.mean(pixel_de[valid]))
        else:
            grand = float(np.mean(pixel_de))

        h, w = pixel_de.shape
        cells = self._grid.get_cell_coords(h, w)
        n_rows = self._grid.rows
        n_cols = self._grid.cols

        row_avg = np.zeros(n_rows, dtype=np.float64)
        col_avg = np.zeros(n_cols, dtype=np.float64)
        cell_avg = np.zeros(len(cells), dtype=np.float64)

        for i, (cy, cx, ch, cw) in enumerate(cells):
            cell_de = pixel_de[cy : cy + ch, cx : cx + cw]
            if mask is not None:
                cell_mask = mask[cy : cy + ch, cx : cx + cw]
                cell_valid = cell_mask > 0
                if np.any(cell_valid):
                    cell_avg[i] = float(np.mean(cell_de[cell_valid]))
                else:
                    cell_avg[i] = np.nan
            else:
                cell_avg[i] = float(np.mean(cell_de))

        cell_avg_grid = cell_avg.reshape(n_rows, n_cols)
        for r in range(n_rows):
            vals = cell_avg_grid[r]
            valid_vals = vals[~np.isnan(vals)]
            row_avg[r] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan
        for c in range(n_cols):
            vals = cell_avg_grid[:, c]
            valid_vals = vals[~np.isnan(vals)]
            col_avg[c] = float(np.mean(valid_vals)) if len(valid_vals) > 0 else np.nan

        return {
            "grand_delta_e": grand,
            "pixel_delta_e": pixel_de,
            "row_avg": row_avg,
            "col_avg": col_avg,
            "cell_avg": cell_avg,
        }
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `pytest tests/test_delta_e.py -v`
Expected: All 5 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add src/core/metrics/base_metric.py src/core/metrics/delta_e.py tests/test_delta_e.py
git commit -m "feat: add base metric class and Delta E metric"
```

---

### Task 9: Contact metric

**Files:**
- Create: `src/core/metrics/contact.py`
- Create: `tests/test_contact.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for contact metric."""

import numpy as np
from src.core.metrics.contact import ContactMetric


class TestContact:
    def test_uniform_image_zero(self):
        """Solid gray image -> contact = 0."""
        metric = ContactMetric(threshold=128)
        gray = np.full((10, 10), 200, dtype=np.uint8)
        ref = gray.copy()
        result = metric.compute(gray, ref)
        assert result["contact_perimeter"] == 0

    def test_half_black_white(self):
        """Left half black, right half white -> contact = vertical boundary."""
        metric = ContactMetric(threshold=128)
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[:, 5:] = 255
        ref = gray.copy()
        result = metric.compute(gray, ref)
        # Vertical boundary: 10 rows × 1 edge each = 10
        assert result["contact_perimeter"] == 10

    def test_checkerboard_high(self):
        """2x2 checkerboard has maximum contact relative to size."""
        metric = ContactMetric(threshold=128)
        gray = np.zeros((4, 4), dtype=np.uint8)
        gray[0, 1] = 255; gray[0, 3] = 255
        gray[1, 0] = 255; gray[1, 2] = 255
        gray[2, 1] = 255; gray[2, 3] = 255
        gray[3, 0] = 255; gray[3, 2] = 255
        ref = gray.copy()
        result = metric.compute(gray, ref)
        assert result["contact_perimeter"] > 0

    def test_mask_excludes_boundary(self):
        """Masked region boundary not counted."""
        metric = ContactMetric(threshold=128)
        gray = np.zeros((10, 10), dtype=np.uint8)
        gray[:, 5:] = 255
        mask = np.ones((10, 10), dtype=np.uint8)
        mask[:, 4:6] = 0  # mask out the boundary columns
        ref = gray.copy()
        result = metric.compute(gray, ref, mask)
        assert result["contact_perimeter"] == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contact.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement contact.py**

```python
"""Contact metric: binary threshold perimeter between light/dark regions."""

from typing import Any

import numpy as np

from src.core.metrics.base_metric import BaseMetric


class ContactMetric(BaseMetric):
    """Counts boundary edges between white and black regions after thresholding."""

    def __init__(self, threshold: int = 128) -> None:
        self._threshold = threshold

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute contact perimeter.

        Args:
            frame: (H, W) grayscale uint8 array.
            reference_frame: Not used (contact is absolute, not relative).
            mask: (H, W) uint8 array, 1=keep, 0=exclude.

        Returns:
            Dict with contact_perimeter.
        """
        binary = (frame >= self._threshold).astype(np.uint8)

        if mask is None:
            mask = np.ones_like(binary, dtype=np.uint8)

        # Horizontal edges: compare pixel (r, c) with (r, c+1)
        h_left = binary[:, :-1]
        h_right = binary[:, 1:]
        m_left = mask[:, :-1]
        m_right = mask[:, 1:]
        h_contact = np.sum((h_left != h_right) & (m_left > 0) & (m_right > 0))

        # Vertical edges: compare pixel (r, c) with (r+1, c)
        v_top = binary[:-1, :]
        v_bottom = binary[1:, :]
        m_top = mask[:-1, :]
        m_bottom = mask[1:, :]
        v_contact = np.sum((v_top != v_bottom) & (m_top > 0) & (m_bottom > 0))

        return {"contact_perimeter": int(h_contact + v_contact)}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contact.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/metrics/contact.py tests/test_contact.py
git commit -m "feat: add contact metric with mask-aware perimeter"
```

---

### Task 10: GLCM builder

**Files:**
- Create: `src/core/metrics/glcm.py`
- Create: `tests/test_glcm.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for GLCM builder."""

import numpy as np
from src.core.metrics.glcm import GLCMBuilder


class TestGLCMBuilder:
    def test_uniform_image(self):
        """Uniform gray -> all weight on diagonal."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((10, 10), 128, dtype=np.uint8)
        glcm = builder.build(gray)
        assert glcm.shape == (16, 16)
        assert abs(np.sum(glcm) - 1.0) < 1e-10  # normalized
        # Quantized level for 128 = 128 * 16 // 256 = 8
        level = 128 * 16 // 256
        assert glcm[level, level] == 1.0

    def test_normalized(self):
        """GLCM sums to 1."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        glcm = builder.build(gray)
        assert abs(np.sum(glcm) - 1.0) < 1e-10

    def test_mask_excludes_pairs(self):
        """Masked pixel pairs are excluded."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 0))  # 1 right
        gray = np.zeros((5, 5), dtype=np.uint8)
        gray[:, 2:] = 255
        mask = np.ones((5, 5), dtype=np.uint8)
        mask[:, 2] = 0  # mask column 2
        glcm = builder.build(gray, mask)
        # Pairs crossing column 2 are excluded (cols 1-2 and 2-3)
        # Only valid pairs: (0,1) all 0->0, and (3,4) all 255->255
        assert abs(np.sum(glcm) - 1.0) < 1e-10
        # Should only have diagonal entries
        off_diag = np.sum(glcm) - np.trace(glcm)
        assert abs(off_diag) < 1e-10

    def test_fully_masked_returns_zeros(self):
        """Fully masked image returns zero GLCM."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((10, 10), 100, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        glcm = builder.build(gray, mask)
        assert np.sum(glcm) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_glcm.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement glcm.py**

```python
"""GLCM (Gray-Level Co-occurrence Matrix) builder."""

import numpy as np


class GLCMBuilder:
    """Builds a normalized GLCM from a grayscale image."""

    def __init__(
        self,
        gray_levels: int = 16,
        offset: tuple[int, int] = (1, 1),
    ) -> None:
        self._gray_levels = gray_levels
        self._dx, self._dy = offset

    @property
    def gray_levels(self) -> int:
        return self._gray_levels

    def build(
        self,
        grayscale: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Build a normalized GLCM from a grayscale image.

        Args:
            grayscale: (H, W) uint8 grayscale image.
            mask: (H, W) uint8 array, 1=keep, 0=exclude.

        Returns:
            (gray_levels, gray_levels) float64 normalized GLCM.
        """
        quantized = (grayscale.astype(np.int32) * self._gray_levels // 256).clip(
            0, self._gray_levels - 1
        )

        h, w = quantized.shape
        dx, dy = self._dx, self._dy

        # Source and target pixel regions
        if dy >= 0:
            src_rows = slice(0, h - dy) if dy > 0 else slice(0, h)
            tgt_rows = slice(dy, h) if dy > 0 else slice(0, h)
        else:
            src_rows = slice(-dy, h)
            tgt_rows = slice(0, h + dy)

        if dx >= 0:
            src_cols = slice(0, w - dx) if dx > 0 else slice(0, w)
            tgt_cols = slice(dx, w) if dx > 0 else slice(0, w)
        else:
            src_cols = slice(-dx, w)
            tgt_cols = slice(0, w + dx)

        src = quantized[src_rows, src_cols].ravel()
        tgt = quantized[tgt_rows, tgt_cols].ravel()

        if mask is not None:
            m_src = mask[src_rows, src_cols].ravel()
            m_tgt = mask[tgt_rows, tgt_cols].ravel()
            valid = (m_src > 0) & (m_tgt > 0)
            src = src[valid]
            tgt = tgt[valid]

        glcm = np.zeros((self._gray_levels, self._gray_levels), dtype=np.float64)

        if len(src) == 0:
            return glcm

        # Vectorized GLCM accumulation
        np.add.at(glcm, (src, tgt), 1)

        total = np.sum(glcm)
        if total > 0:
            glcm /= total

        return glcm
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_glcm.py -v`
Expected: All 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/metrics/glcm.py tests/test_glcm.py
git commit -m "feat: add GLCM builder with mask support"
```

---

### Task 11: Contrast metric

**Files:**
- Create: `src/core/metrics/contrast.py`
- Create: `tests/test_contrast.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for contrast metric."""

import numpy as np
from src.core.metrics.contrast import ContrastMetric
from src.core.metrics.glcm import GLCMBuilder


class TestContrast:
    def test_uniform_zero(self):
        """Uniform image -> GLCM on diagonal -> contrast = 0."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((20, 20), 100, dtype=np.uint8)
        glcm = builder.build(gray)
        metric = ContrastMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert result["contrast"] == 0.0

    def test_checkerboard_high(self):
        """Alternating 0/255 checkerboard -> high contrast."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 0))
        gray = np.zeros((20, 20), dtype=np.uint8)
        gray[:, 1::2] = 255
        glcm = builder.build(gray)
        metric = ContrastMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert result["contrast"] > 100
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_contrast.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement contrast.py**

```python
"""Contrast metric from GLCM."""

from typing import Any

import numpy as np

from src.core.metrics.base_metric import BaseMetric


class ContrastMetric(BaseMetric):
    """Computes contrast from a precomputed normalized GLCM."""

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: np.ndarray | None = None,
        glcm: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute contrast: sum of |i-j|^2 * p_ij.

        Args:
            frame: Not directly used (GLCM precomputed).
            reference_frame: Not used.
            mask: Not used (mask applied during GLCM building).
            glcm: Precomputed normalized GLCM matrix.

        Returns:
            Dict with contrast value.
        """
        if glcm is None:
            raise ValueError("ContrastMetric requires a precomputed GLCM")

        n = glcm.shape[0]
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        contrast = float(np.sum((i - j) ** 2 * glcm))

        return {"contrast": contrast}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_contrast.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/metrics/contrast.py tests/test_contrast.py
git commit -m "feat: add contrast metric from GLCM"
```

---

### Task 12: Homogeneity metric

**Files:**
- Create: `src/core/metrics/homogeneity.py`
- Create: `tests/test_homogeneity.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for homogeneity metric."""

import numpy as np
from src.core.metrics.homogeneity import HomogeneityMetric
from src.core.metrics.glcm import GLCMBuilder


class TestHomogeneity:
    def test_uniform_max(self):
        """Uniform image -> GLCM on diagonal -> homogeneity = 1.0."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((20, 20), 100, dtype=np.uint8)
        glcm = builder.build(gray)
        metric = HomogeneityMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert abs(result["homogeneity"] - 1.0) < 1e-10

    def test_noisy_lower(self):
        """Random noise -> low homogeneity."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 0))
        np.random.seed(42)
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        glcm = builder.build(gray)
        metric = HomogeneityMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert result["homogeneity"] < 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_homogeneity.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement homogeneity.py**

```python
"""Homogeneity metric from GLCM."""

from typing import Any

import numpy as np

from src.core.metrics.base_metric import BaseMetric


class HomogeneityMetric(BaseMetric):
    """Computes homogeneity from a precomputed normalized GLCM."""

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: np.ndarray | None = None,
        glcm: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute homogeneity: sum of p_ij / (1 + |i-j|).

        Args:
            frame: Not directly used.
            reference_frame: Not used.
            mask: Not used (mask applied during GLCM building).
            glcm: Precomputed normalized GLCM matrix.

        Returns:
            Dict with homogeneity value.
        """
        if glcm is None:
            raise ValueError("HomogeneityMetric requires a precomputed GLCM")

        n = glcm.shape[0]
        i, j = np.meshgrid(np.arange(n), np.arange(n), indexing="ij")
        homogeneity = float(np.sum(glcm / (1 + np.abs(i - j))))

        return {"homogeneity": homogeneity}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_homogeneity.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/metrics/homogeneity.py tests/test_homogeneity.py
git commit -m "feat: add homogeneity metric from GLCM"
```

---

### Task 13: Energy metric

**Files:**
- Create: `src/core/metrics/energy.py`
- Create: `tests/test_energy.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for energy (ASM) metric."""

import numpy as np
from src.core.metrics.energy import EnergyMetric
from src.core.metrics.glcm import GLCMBuilder


class TestEnergy:
    def test_uniform_max(self):
        """Uniform image -> single GLCM entry = 1.0 -> energy = 1.0."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 1))
        gray = np.full((20, 20), 100, dtype=np.uint8)
        glcm = builder.build(gray)
        metric = EnergyMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert abs(result["energy"] - 1.0) < 1e-10

    def test_random_low(self):
        """Random noise -> energy close to 0."""
        builder = GLCMBuilder(gray_levels=16, offset=(1, 0))
        np.random.seed(42)
        gray = np.random.randint(0, 256, (50, 50), dtype=np.uint8)
        glcm = builder.build(gray)
        metric = EnergyMetric()
        result = metric.compute(gray, gray, glcm=glcm)
        assert result["energy"] < 0.1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_energy.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement energy.py**

```python
"""Energy (Angular Second Moment / ASM) metric from GLCM."""

from typing import Any

import numpy as np

from src.core.metrics.base_metric import BaseMetric


class EnergyMetric(BaseMetric):
    """Computes energy / ASM from a precomputed normalized GLCM."""

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: np.ndarray | None = None,
        glcm: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute energy: sum of p_ij^2.

        Args:
            frame: Not directly used.
            reference_frame: Not used.
            mask: Not used (mask applied during GLCM building).
            glcm: Precomputed normalized GLCM matrix.

        Returns:
            Dict with energy value.
        """
        if glcm is None:
            raise ValueError("EnergyMetric requires a precomputed GLCM")

        energy = float(np.sum(glcm ** 2))

        return {"energy": energy}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_energy.py -v`
Expected: All 2 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/metrics/energy.py tests/test_energy.py
git commit -m "feat: add energy/ASM metric from GLCM"
```

---

### Task 14: Variance metric

**Files:**
- Create: `src/core/metrics/variance.py`
- Create: `tests/test_variance.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for variance metric."""

import numpy as np
from src.core.metrics.variance import VarianceMetric


class TestVariance:
    def test_uniform_zero(self):
        """Uniform color across all cells -> variance = 0."""
        metric = VarianceMetric(grid_rows=5, grid_cols=5)
        rgb = np.full((100, 100, 3), 128, dtype=np.uint8)
        lab = np.full((100, 100, 3), [50.0, 0.0, 0.0])
        cell_delta_e = np.zeros(25)
        result = metric.compute_variance(rgb, lab, cell_delta_e)
        assert result["variance_r"] == 0.0
        assert result["variance_g"] == 0.0
        assert result["variance_b"] == 0.0
        assert result["variance_l"] == 0.0
        assert result["variance_a"] == 0.0
        assert result["variance_b_star"] == 0.0
        assert result["variance_delta_e"] == 0.0

    def test_different_quadrants_high(self):
        """4 cells with different colors -> high variance."""
        metric = VarianceMetric(grid_rows=2, grid_cols=2)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:50, :50] = [255, 0, 0]
        rgb[:50, 50:] = [0, 255, 0]
        rgb[50:, :50] = [0, 0, 255]
        rgb[50:, 50:] = [255, 255, 0]
        lab = np.random.rand(100, 100, 3) * 50
        cell_delta_e = np.array([10.0, 20.0, 30.0, 40.0])
        result = metric.compute_variance(rgb, lab, cell_delta_e)
        assert result["variance_r"] > 0
        assert result["variance_delta_e"] > 0

    def test_with_mask(self):
        """Masked cells excluded from variance computation."""
        metric = VarianceMetric(grid_rows=2, grid_cols=2)
        rgb = np.zeros((100, 100, 3), dtype=np.uint8)
        rgb[:50, :50] = [255, 0, 0]  # will be fully masked
        rgb[:50, 50:] = [100, 100, 100]
        rgb[50:, :50] = [100, 100, 100]
        rgb[50:, 50:] = [100, 100, 100]
        lab = np.full((100, 100, 3), [50.0, 0.0, 0.0])
        cell_delta_e = np.array([np.nan, 5.0, 5.0, 5.0])
        mask = np.ones((100, 100), dtype=np.uint8)
        mask[:50, :50] = 0
        result = metric.compute_variance(rgb, lab, cell_delta_e, mask)
        # With the red cell masked, remaining cells are all [100,100,100] -> 0 variance
        assert result["variance_r"] == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_variance.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement variance.py**

```python
"""Variance metric: spatial variation of cell-averaged colors."""

from typing import Any

import numpy as np

from src.core.grid_analyzer import GridAnalyzer
from src.core.metrics.base_metric import BaseMetric


class VarianceMetric(BaseMetric):
    """Computes variance of cell-averaged colors across the grid."""

    def __init__(self, grid_rows: int = 5, grid_cols: int = 5) -> None:
        self._grid = GridAnalyzer(rows=grid_rows, cols=grid_cols)

    def compute(
        self,
        frame: np.ndarray,
        reference_frame: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Not called directly by AnalysisEngine — use compute_variance instead.

        Intentional interface deviation: Variance needs precomputed per-cell ΔE
        values from DeltaEMetric, so it can't follow the standard compute() signature.
        AnalysisEngine calls compute_variance() explicitly after Delta E runs.
        """
        raise NotImplementedError("Use compute_variance() with precomputed data")

    def compute_variance(
        self,
        rgb_frame: np.ndarray,
        lab_frame: np.ndarray,
        cell_delta_e: np.ndarray,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Compute variance of cell averages across channels.

        Args:
            rgb_frame: (H, W, 3) uint8 RGB image.
            lab_frame: (H, W, 3) float64 L*a*b* image.
            cell_delta_e: (N_cells,) per-cell average ΔE values.
            mask: (H, W) uint8, 1=keep, 0=exclude.

        Returns:
            Dict with variance_r, variance_g, variance_b,
            variance_l, variance_a, variance_b_star, variance_delta_e.
        """
        rgb_avgs = self._grid.compute_cell_averages(rgb_frame, mask)
        lab_avgs = self._grid.compute_cell_averages(lab_frame, mask)

        h, w = rgb_frame.shape[:2]
        if mask is not None:
            valid = self._grid.get_valid_cells(mask, h, w)
            valid_mask = np.array(valid)
        else:
            valid_mask = np.ones(len(rgb_avgs), dtype=bool)

        # Also exclude cells with NaN delta_e
        valid_de = ~np.isnan(cell_delta_e)
        valid_mask = valid_mask & valid_de

        def _var(arr: np.ndarray) -> float:
            vals = arr[valid_mask]
            if len(vals) < 2:
                return 0.0
            return float(np.var(vals))

        return {
            "variance_r": _var(rgb_avgs[:, 0]),
            "variance_g": _var(rgb_avgs[:, 1]),
            "variance_b": _var(rgb_avgs[:, 2]),
            "variance_l": _var(lab_avgs[:, 0]),
            "variance_a": _var(lab_avgs[:, 1]),
            "variance_b_star": _var(lab_avgs[:, 2]),
            "variance_delta_e": _var(cell_delta_e),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_variance.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/metrics/variance.py tests/test_variance.py
git commit -m "feat: add variance metric with per-channel cell variance"
```

---

### Task 15: Analysis engine

**Files:**
- Create: `src/core/analysis_engine.py`
- Create: `tests/test_analysis_engine.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for analysis engine."""

import numpy as np
import pytest
from unittest.mock import MagicMock
from src.core.analysis_engine import AnalysisEngine
from src.utils.config_loader import DEFAULT_CONFIG


class TestAnalysisEngine:
    def _make_config(self, **overrides):
        config = DEFAULT_CONFIG.copy()
        config.update(overrides)
        return config

    def test_set_reference_frame(self):
        """Reference frame is stored correctly."""
        config = self._make_config()
        engine = AnalysisEngine(config)
        ref = np.full((50, 50, 3), 128, dtype=np.uint8)
        engine.set_reference_frame_data(ref)
        assert engine._reference_frame is not None

    def test_process_single_frame(self):
        """Processing one frame produces one result row."""
        config = self._make_config()
        engine = AnalysisEngine(config)
        ref = np.full((50, 50, 3), 100, dtype=np.uint8)
        frame = np.full((50, 50, 3), 150, dtype=np.uint8)
        engine.set_reference_frame_data(ref)
        engine.process_frame(frame, frame_number=1, timestamp=0.033)
        assert len(engine.results) == 1
        row = engine.results[0]
        assert row["frame_number"] == 1
        assert row["timestamp"] == 0.033
        assert "grand_delta_e" in row
        assert "contact_perimeter" in row
        assert "contrast" in row
        assert "homogeneity" in row
        assert "energy" in row
        assert "variance_r" in row

    def test_glcm_frame_skip(self):
        """When glcm_frame_skip=2, GLCM metrics hold value on non-GLCM frames."""
        config = self._make_config(glcm_frame_skip=2)
        engine = AnalysisEngine(config)
        ref = np.full((50, 50, 3), 100, dtype=np.uint8)
        engine.set_reference_frame_data(ref)
        # Frame 0: uniform -> contrast=0. This is a GLCM frame.
        frame1 = np.full((50, 50, 3), 120, dtype=np.uint8)
        engine.process_frame(frame1, frame_number=0, timestamp=0.0)
        assert engine.results[0]["contrast"] == 0.0
        # Frame 1: half-and-half -> would have high contrast if recomputed.
        # But glcm_frame_skip=2, so GLCM is NOT recomputed. Value held from frame 0.
        frame2 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame2[:, 25:] = 255
        engine.process_frame(frame2, frame_number=1, timestamp=0.033)
        # Contrast should still be 0 (held), not the high value from the split frame
        assert engine.results[1]["contrast"] == 0.0
        # Frame 2: another GLCM frame. Now GLCM is recomputed on a new frame.
        frame3 = np.zeros((50, 50, 3), dtype=np.uint8)
        frame3[:, 25:] = 200
        engine.process_frame(frame3, frame_number=2, timestamp=0.066)
        # Now contrast should be recomputed and be > 0
        assert engine.results[2]["contrast"] > 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_analysis_engine.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement analysis_engine.py**

```python
"""Analysis engine: orchestrates all metrics per frame."""

import logging
import time
from typing import Any

import cv2
import numpy as np

from src.core.frame_processor import FrameProcessor
from src.core.grid_analyzer import GridAnalyzer
from src.core.metrics.contact import ContactMetric
from src.core.metrics.contrast import ContrastMetric
from src.core.metrics.delta_e import DeltaEMetric
from src.core.metrics.energy import EnergyMetric
from src.core.metrics.glcm import GLCMBuilder
from src.core.metrics.homogeneity import HomogeneityMetric
from src.core.metrics.variance import VarianceMetric
from src.utils.color_convert import rgb_to_lab

logger = logging.getLogger("kineticolor")


class AnalysisEngine:
    """Orchestrates per-frame metric computation and time series storage."""

    def __init__(self, config: dict[str, Any]) -> None:
        self._config = config
        self._processor = FrameProcessor(
            brightness_change_threshold=config.get("brightness_change_threshold", 0.2)
        )

        grid_rows = config["grid_rows"]
        grid_cols = config["grid_cols"]

        self._delta_e = DeltaEMetric(grid_rows=grid_rows, grid_cols=grid_cols)
        self._contact = ContactMetric(threshold=config["contact_threshold"])
        self._glcm_builder = GLCMBuilder(
            gray_levels=config["glcm_gray_levels"],
            offset=tuple(config["glcm_offset"]),
        )
        self._contrast = ContrastMetric()
        self._homogeneity = HomogeneityMetric()
        self._energy = EnergyMetric()
        self._variance = VarianceMetric(grid_rows=grid_rows, grid_cols=grid_cols)
        self._grid = GridAnalyzer(rows=grid_rows, cols=grid_cols)

        self._reference_frame: np.ndarray | None = None
        self._reference_lab: np.ndarray | None = None
        self._reference_gray: np.ndarray | None = None

        self._results: list[dict[str, Any]] = []
        self._glcm_frame_skip = config.get("glcm_frame_skip", 1)
        self._analyzed_frame_count = 0
        self._last_glcm_results: dict[str, Any] = {
            "contrast": 0.0,
            "homogeneity": 1.0,
            "energy": 1.0,
        }

    @property
    def results(self) -> list[dict[str, Any]]:
        return self._results

    def set_reference_frame_data(
        self, frame: np.ndarray, mask: np.ndarray | None = None
    ) -> None:
        """Set the reference frame for Delta E computation.

        Args:
            frame: BGR uint8 frame.
            mask: Optional exclusion mask.
        """
        self._reference_frame = frame.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self._reference_lab = rgb_to_lab(rgb)
        self._reference_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        logger.info("Reference frame set")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_number: int,
        timestamp: float,
        roi: tuple[int, int, int, int] | None = None,
        mask: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Process a single frame through all metrics.

        Args:
            frame: BGR uint8 frame.
            frame_number: Zero-based frame index.
            timestamp: Time in seconds.
            roi: (x, y, w, h) region of interest or None.
            mask: (H, W) exclusion mask or None.

        Returns:
            Dict of all metric values for this frame.
        """
        t_start = time.perf_counter()

        if self._reference_frame is None:
            self.set_reference_frame_data(frame, mask)

        # Crop to ROI
        cropped = self._processor.crop_to_roi(frame, roi)
        ref_cropped = self._processor.crop_to_roi(self._reference_frame, roi)

        # Apply mask
        if mask is not None:
            if roi is not None:
                x, y, w, h = roi
                roi_mask = mask[y : y + h, x : x + w].copy()
            else:
                roi_mask = mask
        else:
            roi_mask = None

        masked = self._processor.apply_mask(cropped, roi_mask)
        ref_masked = self._processor.apply_mask(ref_cropped, roi_mask)

        # Brightness check
        self._processor.check_brightness(masked, roi_mask)

        # Color conversions
        rgb = cv2.cvtColor(masked, cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(ref_masked, cv2.COLOR_BGR2RGB)
        lab = rgb_to_lab(rgb)
        ref_lab = rgb_to_lab(ref_rgb)
        gray = cv2.cvtColor(masked, cv2.COLOR_BGR2GRAY)

        # 1. Delta E (must run before Variance)
        t_de = time.perf_counter()
        de_result = self._delta_e.compute(lab, ref_lab, roi_mask)
        logger.debug(f"Frame {frame_number}: Delta E computed in {time.perf_counter() - t_de:.4f}s")

        # 2. Contact
        t_contact = time.perf_counter()
        contact_result = self._contact.compute(gray, None, roi_mask)
        logger.debug(f"Frame {frame_number}: Contact computed in {time.perf_counter() - t_contact:.4f}s")

        # 3. GLCM metrics (with frame skip)
        is_glcm_frame = (self._analyzed_frame_count % self._glcm_frame_skip) == 0
        if is_glcm_frame:
            t_glcm = time.perf_counter()
            glcm = self._glcm_builder.build(gray, roi_mask)
            contrast_result = self._contrast.compute(gray, None, glcm=glcm)
            homogeneity_result = self._homogeneity.compute(gray, None, glcm=glcm)
            energy_result = self._energy.compute(gray, None, glcm=glcm)
            self._last_glcm_results = {
                "contrast": contrast_result["contrast"],
                "homogeneity": homogeneity_result["homogeneity"],
                "energy": energy_result["energy"],
            }
            logger.debug(f"Frame {frame_number}: GLCM metrics computed in {time.perf_counter() - t_glcm:.4f}s")
        else:
            logger.debug(f"Frame {frame_number}: GLCM metrics held from previous frame")

        # 4. Variance (needs cell Delta E from step 1)
        t_var = time.perf_counter()
        var_result = self._variance.compute_variance(
            rgb, lab, de_result["cell_avg"], roi_mask
        )
        logger.debug(f"Frame {frame_number}: Variance computed in {time.perf_counter() - t_var:.4f}s")

        # Assemble result row
        row: dict[str, Any] = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "grand_delta_e": de_result["grand_delta_e"],
            "contact_perimeter": contact_result["contact_perimeter"],
            "contrast": self._last_glcm_results["contrast"],
            "homogeneity": self._last_glcm_results["homogeneity"],
            "energy": self._last_glcm_results["energy"],
        }
        row.update(var_result)

        self._results.append(row)
        self._analyzed_frame_count += 1

        t_total = time.perf_counter() - t_start
        logger.debug(f"Frame {frame_number}: Total processing time {t_total:.4f}s")

        return row
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_analysis_engine.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/analysis_engine.py tests/test_analysis_engine.py
git commit -m "feat: add analysis engine orchestrating all metrics"
```

---

### Task 16: Data exporter

**Files:**
- Create: `src/core/export.py`
- Create: `tests/test_export.py`

- [ ] **Step 1: Write failing tests**

```python
"""Tests for data exporter."""

import csv
from pathlib import Path
import pytest
from src.core.export import DataExporter


SAMPLE_RESULTS = [
    {
        "frame_number": 0, "timestamp": 0.0,
        "grand_delta_e": 0.0, "contact_perimeter": 0,
        "contrast": 0.0, "homogeneity": 1.0, "energy": 1.0,
        "variance_r": 0.0, "variance_g": 0.0, "variance_b": 0.0,
        "variance_l": 0.0, "variance_a": 0.0, "variance_b_star": 0.0,
        "variance_delta_e": 0.0,
    },
    {
        "frame_number": 1, "timestamp": 0.033,
        "grand_delta_e": 5.2, "contact_perimeter": 42,
        "contrast": 3.1, "homogeneity": 0.8, "energy": 0.6,
        "variance_r": 10.0, "variance_g": 8.0, "variance_b": 12.0,
        "variance_l": 5.0, "variance_a": 2.0, "variance_b_star": 3.0,
        "variance_delta_e": 4.5,
    },
]


class TestDataExporter:
    def test_export_csv(self, tmp_path):
        """CSV export creates valid file with correct rows."""
        out = tmp_path / "results.csv"
        exporter = DataExporter()
        exporter.export(SAMPLE_RESULTS, out, fmt="csv")
        assert out.exists()
        with open(out) as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["frame_number"] == "0"
        assert rows[1]["grand_delta_e"] == "5.2"

    def test_export_xlsx(self, tmp_path):
        """XLSX export creates a valid file."""
        out = tmp_path / "results.xlsx"
        exporter = DataExporter()
        exporter.export(SAMPLE_RESULTS, out, fmt="xlsx")
        assert out.exists()
        assert out.stat().st_size > 0

    def test_empty_results(self, tmp_path):
        """Exporting empty results still creates file with header."""
        out = tmp_path / "empty.csv"
        exporter = DataExporter()
        exporter.export([], out, fmt="csv")
        assert out.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_export.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement export.py**

```python
"""Data exporter: CSV and XLSX output for time series results."""

import csv
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("kineticolor")

COLUMNS = [
    "frame_number", "timestamp",
    "grand_delta_e", "contact_perimeter",
    "contrast", "homogeneity", "energy",
    "variance_r", "variance_g", "variance_b",
    "variance_l", "variance_a", "variance_b_star",
    "variance_delta_e",
]


class DataExporter:
    """Exports analysis results to CSV or XLSX."""

    def export(
        self,
        results: list[dict[str, Any]],
        output_path: Path | str,
        fmt: str = "csv",
    ) -> None:
        """Export results to file.

        Args:
            results: List of per-frame result dicts.
            output_path: Output file path.
            fmt: 'csv' or 'xlsx'.
        """
        output_path = Path(output_path)

        if fmt == "csv":
            self._export_csv(results, output_path)
        elif fmt == "xlsx":
            self._export_xlsx(results, output_path)
        else:
            raise ValueError(f"Unsupported format: {fmt}")

        logger.info(f"Exported {len(results)} rows to {output_path}")

    def _export_csv(
        self, results: list[dict[str, Any]], path: Path
    ) -> None:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for row in results:
                writer.writerow(row)

    def _export_xlsx(
        self, results: list[dict[str, Any]], path: Path
    ) -> None:
        from openpyxl import Workbook

        wb = Workbook()
        ws = wb.active
        ws.title = "Metrics"
        ws.append(COLUMNS)
        for row in results:
            ws.append([row.get(col) for col in COLUMNS])
        wb.save(path)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_export.py -v`
Expected: All 3 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/core/export.py tests/test_export.py
git commit -m "feat: add data exporter with CSV and XLSX support"
```

---

### Task 17: CLI entry point

**Files:**
- Create: `src/main.py`

- [ ] **Step 1: Implement main.py**

```python
"""CLI entry point for Kineticolor analysis."""

import argparse
import sys
from pathlib import Path

from src.core.analysis_engine import AnalysisEngine
from src.core.export import DataExporter
from src.core.video_reader import VideoReader
from src.utils.config_loader import load_config
from src.utils.logger import setup_logger


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Kineticolor: Computer vision mixing analysis"
    )
    parser.add_argument(
        "--video", type=str, required=True, help="Path to video file"
    )
    parser.add_argument(
        "--roi", type=str, default=None,
        help="ROI as x,y,w,h (default: full frame)"
    )
    parser.add_argument(
        "--config", type=str, default="config/default_config.yaml",
        help="Path to config YAML"
    )
    parser.add_argument(
        "--output", type=str, default="results.csv",
        help="Output file path"
    )
    parser.add_argument(
        "--reference-frame", type=int, default=0,
        help="Reference frame number (default: 0)"
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logger = setup_logger()

    config_path = Path(args.config)
    if config_path.exists():
        config = load_config(config_path)
    else:
        logger.warning(f"Config not found at {config_path}, using defaults")
        config = load_config()

    roi = None
    if args.roi:
        parts = [int(x.strip()) for x in args.roi.split(",")]
        if len(parts) != 4:
            logger.error("ROI must be x,y,w,h")
            sys.exit(1)
        roi = tuple(parts)

    reader = VideoReader(
        path=args.video,
        frame_skip=config["frame_skip"],
        fps_override=config.get("video_fps_override"),
    )

    engine = AnalysisEngine(config)

    # Set reference frame
    if args.reference_frame > 0:
        ref = reader.get_frame(args.reference_frame)
        if ref is None:
            logger.error(f"Cannot read reference frame {args.reference_frame}")
            sys.exit(1)
        engine.set_reference_frame_data(ref)
        logger.info(f"Using frame {args.reference_frame} as reference")

    logger.info("Starting analysis...")
    for frame_number, frame in reader:
        timestamp = reader.timestamp(frame_number)
        engine.process_frame(frame, frame_number, timestamp, roi=roi)

    reader.release()

    # Export
    output_path = Path(args.output)
    fmt = config.get("export_format", "csv")
    if output_path.suffix == ".xlsx":
        fmt = "xlsx"
    elif output_path.suffix == ".csv":
        fmt = "csv"

    exporter = DataExporter()
    exporter.export(engine.results, output_path, fmt=fmt)
    logger.info(f"Analysis complete. {len(engine.results)} frames processed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify import works**

Run: `python -c "from src.main import parse_args; print(parse_args(['--video', 'test.mp4']))"`
Expected: Prints the parsed namespace.

- [ ] **Step 3: Commit**

```bash
git add src/main.py
git commit -m "feat: add CLI entry point for headless analysis"
```

---

### Task 18: Full integration test

**Files:**
- Create: `tests/test_integration.py`

- [ ] **Step 1: Write integration test with synthetic video**

```python
"""Integration test: end-to-end analysis on a synthetic video."""

import cv2
import numpy as np
from pathlib import Path
from src.core.analysis_engine import AnalysisEngine
from src.core.export import DataExporter
from src.utils.config_loader import load_config


def _create_synthetic_video(path: Path, n_frames: int = 10) -> None:
    """Create a simple video that transitions from blue to green."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, 30.0, (100, 100))
    for i in range(n_frames):
        t = i / (n_frames - 1)
        color = np.array([255 * (1 - t), 255 * t, 0], dtype=np.uint8)  # BGR
        frame = np.full((100, 100, 3), color, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class TestIntegration:
    def test_end_to_end(self, tmp_path):
        """Full pipeline: synthetic video -> analysis -> CSV export."""
        video_path = tmp_path / "test.mp4"
        _create_synthetic_video(video_path, n_frames=5)

        config = load_config()
        engine = AnalysisEngine(config)

        cap = cv2.VideoCapture(str(video_path))
        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            engine.process_frame(frame, frame_num, frame_num / 30.0)
            frame_num += 1
        cap.release()

        assert len(engine.results) == 5
        # First frame is reference -> delta_e should be 0
        assert engine.results[0]["grand_delta_e"] == 0.0
        # Later frames should have increasing delta_e
        assert engine.results[-1]["grand_delta_e"] > engine.results[1]["grand_delta_e"]

        # Export
        csv_path = tmp_path / "results.csv"
        exporter = DataExporter()
        exporter.export(engine.results, csv_path, fmt="csv")
        assert csv_path.exists()

        xlsx_path = tmp_path / "results.xlsx"
        exporter.export(engine.results, xlsx_path, fmt="xlsx")
        assert xlsx_path.exists()
```

- [ ] **Step 2: Run integration test**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Run full test suite**

Run: `pytest tests/ -v`
Expected: All tests PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add end-to-end integration test with synthetic video"
```
