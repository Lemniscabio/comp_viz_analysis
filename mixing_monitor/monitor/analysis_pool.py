"""ThreadPoolExecutor coordinator for parallel vessel analysis."""

from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor

import numpy as np

from ..common.vessel_analyzer import VesselAnalyzer

logger = logging.getLogger(__name__)


class AnalysisPool:
    """Manages parallel analysis of up to 4 vessel crops."""

    def __init__(self, max_workers: int = 4) -> None:
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="analysis")
        self._analyzers: dict[int, VesselAnalyzer] = {}

    def set_reference(self, vessel_id: int, reference_crop: np.ndarray) -> None:
        """Set/reset the reference frame for a vessel."""
        self._analyzers[vessel_id] = VesselAnalyzer(reference_crop)
        logger.debug("Reference set for vessel %d", vessel_id)

    def get_reference_mean_a(self, vessel_id: int) -> float | None:
        """Return the reference mean a* for a vessel, or None if not set."""
        analyzer = self._analyzers.get(vessel_id)
        return analyzer.reference_mean_a if analyzer is not None else None

    def get_reference_pink_fraction(self, vessel_id: int) -> float | None:
        """Return the reference pink fraction for a vessel, or None if not set."""
        analyzer = self._analyzers.get(vessel_id)
        return analyzer.reference_pink_fraction if analyzer is not None else None

    def submit_all(self, crops: dict[int, np.ndarray]) -> dict[int, Future]:
        """Submit crops for all vessels that have an analyzer. Returns {vessel_id: Future}."""
        futures: dict[int, Future] = {}
        for vid, crop in crops.items():
            if vid in self._analyzers:
                futures[vid] = self._pool.submit(self._analyzers[vid].analyze, crop)
            else:
                logger.debug("No analyzer for vessel %d — skipping", vid)
        return futures

    def remove(self, vessel_id: int) -> None:
        """Remove the analyzer for a vessel (e.g., on reset)."""
        self._analyzers.pop(vessel_id, None)

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
