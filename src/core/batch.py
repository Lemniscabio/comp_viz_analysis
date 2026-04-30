"""Batch analysis: per-video summary CSV and JSON config dump."""
from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional, Tuple

from src.core.mixing_time import MixingTimeParams, MixingTimeResult

BATCH_CSV_COLUMNS = [
    "video_file", "status", "confidence", "fps", "duration_s", "frame_count",
    "roi_x", "roi_y", "roi_w", "roi_h", "t_start_s",
    "visual_t_mix_s", "delta_t_mix_95_s", "abs_delta_t_mix_95_s",
    "t_deltaE_90_s", "t_deltaE_95_s", "t_deltaE_99_s",
    "t_cell_90_s", "t_cell_95_s", "t_cell_99_s",
    "t_variance_90_s", "t_variance_95_s", "t_variance_99_s",
    "t_spatial_90_s", "t_spatial_95_s", "t_spatial_99_s",
    "t_contact_90_s", "t_contact_95_s", "t_contact_99_s",
    "t_contrast_90_s", "t_contrast_95_s", "t_contrast_99_s",
    "t_texture_90_s", "t_texture_95_s", "t_texture_99_s",
    "t_mix_90_s", "t_mix_95_s", "t_mix_99_s",
    "grand_deltaE_amplitude", "contact_amplitude", "contrast_amplitude",
    "final_tail_slope_deltaE", "final_tail_slope_contact",
    "final_tail_slope_contrast", "notes",
]


def _g(d, k):
    """Format dict[k] as a CSV-friendly string. NaN/None -> empty string."""
    v = d.get(k)
    if v is None:
        return ""
    try:
        if v != v:  # NaN
            return ""
        return f"{float(v):.4f}"
    except (TypeError, ValueError):
        return ""


def _row_for(video_file, fps, duration_s, frame_count, roi, result: MixingTimeResult,
             visual_t: float = float("nan")):
    if roi is not None:
        rx, ry, rw, rh = roi
    else:
        rx = ry = rw = rh = ""

    t95 = result.t_mix.get(0.95, float("nan"))
    if visual_t == visual_t and t95 == t95:  # both finite
        delta = t95 - visual_t
        visual_str = f"{visual_t:.4f}"
        delta_str = f"{delta:.4f}"
        abs_delta_str = f"{abs(delta):.4f}"
    else:
        visual_str = f"{visual_t:.4f}" if visual_t == visual_t else ""
        delta_str = ""
        abs_delta_str = ""

    row = {
        "video_file": video_file,
        "status": result.status,
        "confidence": result.confidence,
        "fps": f"{fps:.3f}" if fps else "",
        "duration_s": f"{duration_s:.3f}" if duration_s else "",
        "frame_count": frame_count,
        "roi_x": rx, "roi_y": ry, "roi_w": rw, "roi_h": rh,
        "t_start_s": f"{result.t_start_s:.4f}",
        "visual_t_mix_s": visual_str,
        "delta_t_mix_95_s": delta_str,
        "abs_delta_t_mix_95_s": abs_delta_str,
    }
    levels = (0.90, 0.95, 0.99)
    suffixes = ("90", "95", "99")
    for L, sfx in zip(levels, suffixes):
        row[f"t_deltaE_{sfx}_s"]   = _g(result.t_deltaE,   L)
        row[f"t_cell_{sfx}_s"]     = _g(result.t_cell,     L)
        row[f"t_variance_{sfx}_s"] = _g(result.t_variance, L)
        row[f"t_spatial_{sfx}_s"]  = _g(result.t_spatial,  L)
        row[f"t_contact_{sfx}_s"]  = _g(result.t_contact,  L)
        row[f"t_contrast_{sfx}_s"] = _g(result.t_contrast, L)
        row[f"t_texture_{sfx}_s"]  = _g(result.t_texture,  L)
        row[f"t_mix_{sfx}_s"]      = _g(result.t_mix,      L)
    row["grand_deltaE_amplitude"] = _g(result.amplitudes, "grand_delta_e")
    row["contact_amplitude"]      = _g(result.amplitudes, "contact")
    row["contrast_amplitude"]     = _g(result.amplitudes, "contrast")
    row["final_tail_slope_deltaE"]   = _g(result.tail_slopes, "grand_delta_e")
    row["final_tail_slope_contact"]  = _g(result.tail_slopes, "contact")
    row["final_tail_slope_contrast"] = _g(result.tail_slopes, "contrast")
    row["notes"] = " | ".join(result.notes)
    return row


def write_summary_row(
    csv_path: Path,
    *,
    video_file: str,
    fps: float,
    duration_s: float,
    frame_count: int,
    roi: Optional[Tuple[int, int, int, int]],
    result: MixingTimeResult,
    visual_t: float = float("nan"),
    append: bool = True,
) -> None:
    """Append (or create) one row in the batch summary CSV."""
    csv_path = Path(csv_path)
    write_header = not append or not csv_path.exists()
    mode = "a" if append and csv_path.exists() else "w"
    with open(csv_path, mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=BATCH_CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(_row_for(
            video_file, fps, duration_s, frame_count, roi, result, visual_t,
        ))


def write_batch_config(
    out_path: Path,
    *,
    roi: Optional[Tuple[int, int, int, int]],
    mask_present: bool,
    grid: Tuple[int, int],
    params: Optional[MixingTimeParams],
) -> None:
    """Dump the batch configuration as JSON next to the summary CSV."""
    payload = {
        "roi": list(roi) if roi else None,
        "mask_present": mask_present,
        "grid_rows": grid[0],
        "grid_cols": grid[1],
        "mixing_params": asdict(params) if params else None,
    }
    Path(out_path).write_text(json.dumps(payload, indent=2))
