"""Mixing-time level crossings — lifted verbatim from main's
src/gui/plots_panel.py:_update_mixing_marker. NOTHING from feat/mixing-time.

mixing time at level L = first timestamp where (grand ΔE / max grand ΔE) >= L.
"""
from __future__ import annotations
from typing import Dict, List, Optional, Sequence


def normalized_delta_e(grand: Sequence[float]) -> List[float]:
    m = max(grand) if grand and max(grand) > 0 else 1.0
    return [g / m for g in grand]


def level_times(
    timestamps: Sequence[float],
    grand_delta_e: Sequence[float],
    levels: Sequence[float] = (0.90, 0.95, 0.99),
) -> Dict[float, Optional[float]]:
    norm = normalized_delta_e(grand_delta_e)
    out: Dict[float, Optional[float]] = {}
    for L in levels:
        idx = next((i for i, v in enumerate(norm) if v >= L), None)  # == np.argmax + guard
        out[L] = float(timestamps[idx]) if idx is not None else None
    return out
