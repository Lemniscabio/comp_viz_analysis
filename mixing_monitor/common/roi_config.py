"""Load, save, and validate vessel_rois.json config files."""

import json
import logging
from pathlib import Path

from .constants import (
    CONFIG_SCHEMA_VERSION,
    DEFAULT_CONFIG_PATH,
    MAX_VESSELS,
    MIN_VESSELS,
    VESSEL_COLORS,
    VESSEL_LABELS,
)

logger = logging.getLogger(__name__)


class ConfigValidationError(ValueError):
    """Raised when the config JSON fails schema validation."""


def _validate(data: dict) -> None:
    """Raise ConfigValidationError if data does not match the expected schema."""
    if not isinstance(data, dict):
        raise ConfigValidationError("Root must be a JSON object")

    if data.get("version") != CONFIG_SCHEMA_VERSION:
        raise ConfigValidationError(
            f"Unsupported config version: {data.get('version')}. Expected {CONFIG_SCHEMA_VERSION}"
        )

    cam = data.get("camera")
    if not isinstance(cam, dict):
        raise ConfigValidationError("Missing or invalid 'camera' object")
    for field in ("device_index", "device_name", "resolution"):
        if field not in cam:
            raise ConfigValidationError(f"camera.{field} is required")
    if not (isinstance(cam["resolution"], list) and len(cam["resolution"]) == 2):
        raise ConfigValidationError("camera.resolution must be [width, height]")

    vc = data.get("vessel_count")
    if not isinstance(vc, int) or not (MIN_VESSELS <= vc <= MAX_VESSELS):
        raise ConfigValidationError(
            f"vessel_count must be an integer between {MIN_VESSELS} and {MAX_VESSELS}"
        )

    vessels = data.get("vessels")
    if not isinstance(vessels, list) or len(vessels) != vc:
        raise ConfigValidationError(
            f"vessels array must have exactly {vc} entries (vessel_count={vc})"
        )

    for v in vessels:
        if not isinstance(v, dict):
            raise ConfigValidationError("Each vessel entry must be an object")
        for field in ("id", "label", "color", "roi"):
            if field not in v:
                raise ConfigValidationError(f"Vessel entry missing field '{field}'")
        if not (isinstance(v["roi"], list) and len(v["roi"]) == 4):
            raise ConfigValidationError("roi must be [x, y, width, height]")
        if not all(isinstance(n, (int, float)) for n in v["roi"]):
            raise ConfigValidationError("roi values must be numbers")
        if v["roi"][2] <= 0 or v["roi"][3] <= 0:
            raise ConfigValidationError("roi width and height must be positive")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> dict:
    """Load and validate a vessel_rois.json file.

    Raises:
        FileNotFoundError: if path does not exist.
        ConfigValidationError: if the JSON fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ConfigValidationError(f"Invalid JSON: {exc}") from exc
    _validate(data)
    logger.info("Loaded config from %s (%d vessels)", path, data["vessel_count"])
    return data


def save_config(data: dict, path: Path | str = DEFAULT_CONFIG_PATH) -> None:
    """Validate and write config to path.  Creates parent directories."""
    _validate(data)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    logger.info("Saved config to %s", path)


def make_default_config(
    device_index: int = 0,
    device_name: str = "Camera 0",
    resolution: tuple[int, int] = (1280, 720),
    vessel_count: int = 3,
) -> dict:
    """Return a skeleton config dict (no ROIs — they must be filled in)."""
    vessels = [
        {
            "id": i,
            "label": VESSEL_LABELS[i],
            "color": VESSEL_COLORS[i],
            "roi": [0, 0, 200, 200],  # placeholder — must be replaced
        }
        for i in range(1, vessel_count + 1)
    ]
    return {
        "version": CONFIG_SCHEMA_VERSION,
        "camera": {
            "device_index": device_index,
            "device_name": device_name,
            "resolution": list(resolution),
        },
        "vessel_count": vessel_count,
        "vessels": vessels,
    }


def scale_rois(data: dict, new_resolution: tuple[int, int]) -> dict:
    """Return a copy of data with all ROIs scaled to new_resolution.

    Used when the monitor app's camera resolution differs from calibration.
    """
    import copy

    old_w, old_h = data["camera"]["resolution"]
    new_w, new_h = new_resolution
    if (old_w, old_h) == (new_w, new_h):
        return data

    sx = new_w / old_w
    sy = new_h / old_h

    scaled = copy.deepcopy(data)
    for vessel in scaled["vessels"]:
        x, y, w, h = vessel["roi"]
        vessel["roi"] = [
            round(x * sx),
            round(y * sy),
            round(w * sx),
            round(h * sy),
        ]
    scaled["camera"]["resolution"] = [new_w, new_h]
    logger.info(
        "Scaled ROIs from %dx%d to %dx%d (sx=%.3f, sy=%.3f)",
        old_w, old_h, new_w, new_h, sx, sy,
    )
    return scaled
