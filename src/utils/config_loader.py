"""YAML config loader with defaults and validation."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, Union

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


def load_config(path: Optional[Union[Path, str]] = None) -> dict[str, Any]:
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
