"""Analysis engine: orchestrates all metrics per frame."""
from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional, Tuple

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

    def __init__(self, config: Dict[str, Any]) -> None:
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

        self._reference_frame: Optional[np.ndarray] = None
        self._reference_lab: Optional[np.ndarray] = None
        self._reference_gray: Optional[np.ndarray] = None

        self._results: List[Dict[str, Any]] = []
        self._glcm_frame_skip = config.get("glcm_frame_skip", 1)
        self._analyzed_frame_count = 0
        self._last_glcm_results: Dict[str, Any] = {
            "contrast": 0.0, "homogeneity": 1.0, "energy": 1.0,
        }

    @property
    def results(self) -> List[Dict[str, Any]]:
        return self._results

    def set_reference_frame_data(self, frame: np.ndarray, mask: Optional[np.ndarray] = None) -> None:
        """Set the reference frame for Delta E computation. Frame is BGR uint8."""
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
        roi: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Process a single BGR frame through all metrics.

        Args:
            frame: (H, W, 3) BGR uint8 image.
            frame_number: Frame index in the video/stream.
            timestamp: Time in seconds from the start of analysis.
            roi: Optional (x, y, w, h) region of interest. None = full frame.
            mask: Optional (H, W) uint8 exclusion mask (1=keep, 0=exclude).

        Returns:
            Dict of metric values for this frame (also appended to self.results).
        """
        t_start = time.perf_counter()

        if self._reference_frame is None:
            self.set_reference_frame_data(frame, mask)

        # Crop to ROI
        cropped = self._processor.crop_to_roi(frame, roi)
        ref_cropped = self._processor.crop_to_roi(self._reference_frame, roi)

        # Crop mask to ROI if provided
        if mask is not None:
            if roi is not None:
                x, y, w, h = roi
                roi_mask = mask[y:y + h, x:x + w].copy()
            else:
                roi_mask = mask
        else:
            roi_mask = None

        # Brightness change check (on raw cropped frame, mask-aware)
        self._processor.check_brightness(cropped, roi_mask)

        # Color space conversions on raw pixels (mask NOT applied to pixel data —
        # each metric handles masking internally via the roi_mask parameter)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        ref_rgb = cv2.cvtColor(ref_cropped, cv2.COLOR_BGR2RGB)
        lab = rgb_to_lab(rgb)
        ref_lab = rgb_to_lab(ref_rgb)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # 1. Delta E (must run before Variance — provides cell_avg)
        t_de = time.perf_counter()
        de_result = self._delta_e.compute(lab, ref_lab, roi_mask)
        logger.debug(
            f"Frame {frame_number}: Delta E computed in {time.perf_counter() - t_de:.4f}s"
        )

        # 2. Contact
        t_contact = time.perf_counter()
        contact_result = self._contact.compute(gray, None, roi_mask)
        logger.debug(
            f"Frame {frame_number}: Contact computed in {time.perf_counter() - t_contact:.4f}s"
        )

        # 3. GLCM metrics (with configurable frame skip)
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
            logger.debug(
                f"Frame {frame_number}: GLCM metrics computed in "
                f"{time.perf_counter() - t_glcm:.4f}s"
            )
        else:
            logger.debug(f"Frame {frame_number}: GLCM metrics held from previous frame")

        # 4. Variance (needs cell Delta E from step 1)
        t_var = time.perf_counter()
        var_result = self._variance.compute_variance(rgb, lab, de_result["cell_avg"], roi_mask)
        logger.debug(
            f"Frame {frame_number}: Variance computed in {time.perf_counter() - t_var:.4f}s"
        )

        # Assemble stored row (no large arrays — saved in time series)
        stored_row: Dict[str, Any] = {
            "frame_number": frame_number,
            "timestamp": timestamp,
            "grand_delta_e": de_result["grand_delta_e"],
            "contact_perimeter": contact_result["contact_perimeter"],
            "contrast": self._last_glcm_results["contrast"],
            "homogeneity": self._last_glcm_results["homogeneity"],
            "energy": self._last_glcm_results["energy"],
        }
        stored_row.update(var_result)
        self._results.append(stored_row)

        # Return full result including transient data for GUI
        full_row = dict(stored_row)
        full_row["pixel_delta_e"] = de_result["pixel_delta_e"]
        full_row["row_avg"] = de_result["row_avg"]
        full_row["col_avg"] = de_result["col_avg"]
        self._analyzed_frame_count += 1

        t_total = time.perf_counter() - t_start
        logger.debug(f"Frame {frame_number}: Total processing time {t_total:.4f}s")
        return full_row
