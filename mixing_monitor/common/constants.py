"""Shared constants for the mixing monitor system."""

from pathlib import Path

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

MIXING_MONITOR_DIR = Path(__file__).parent.parent
CONFIG_DIR = MIXING_MONITOR_DIR / "config"
RESULTS_DIR = MIXING_MONITOR_DIR / "results"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "vessel_rois.json"

# ---------------------------------------------------------------------------
# Config schema
# ---------------------------------------------------------------------------

CONFIG_SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Vessel color mapping (fixed per ID, matches physical vessel cap colors)
# ---------------------------------------------------------------------------

VESSEL_COLORS: dict[int, str] = {
    1: "#ef4444",  # red
    2: "#22c55e",  # green
    3: "#3b82f6",  # blue
    4: "#eab308",  # yellow
}

VESSEL_LABELS: dict[int, str] = {
    1: "V1",
    2: "V2",
    3: "V3",
    4: "V4",
}

MAX_VESSELS = 4
MIN_VESSELS = 1

# ---------------------------------------------------------------------------
# ROI constraints
# ---------------------------------------------------------------------------

MIN_ROI_WIDTH = 50   # pixels (camera coordinates)
MIN_ROI_HEIGHT = 50  # pixels (camera coordinates)

# ---------------------------------------------------------------------------
# Camera defaults
# ---------------------------------------------------------------------------

DEFAULT_RESOLUTION = (1280, 720)
CAMERA_PROBE_RANGE = range(10)  # indices 0-9 to probe when pygrabber unavailable

# ---------------------------------------------------------------------------
# Analysis thresholds
# ---------------------------------------------------------------------------

# --- Mixing trigger ---
DELTA_E_TRIGGER_THRESHOLD: float = 3.0  # Grand ΔE must exceed this to enter Mixing state

# --- ΔE stability criterion (generic — works regardless of final color direction) ---
# Mixing is complete when ΔE stops changing, not when it reaches a specific level.
# Rolling std of ΔE over a window: low = color has stabilized = mixing done.
DELTA_E_STABILITY_WINDOW: int = 30          # frames in rolling window for std computation
DELTA_E_STABILITY_THRESHOLD: float = 1.5    # rolling std must drop below this (ΔE units)

# --- Pink pixel fraction criterion (pH titration with phenolphthalein or similar) ---
# Pink (high a*) pixels disappear as HCl neutralizes phenolphthalein.
# Only applied when pink was detected at arm time or during mixing.
PINK_A_STAR_THRESHOLD: float = 10.0     # Lab a* above which a pixel counts as "pink"
PINK_FRACTION_TRIGGER: float = 0.10     # fraction above which solution is considered "pink"
PINK_FRACTION_COMPLETE: float = 0.02    # pink fraction must drop below this to complete

# --- Combined completion: stability AND pink must both hold simultaneously ---
COMPLETION_CONFIRMATION_FRAMES: int = 15    # consecutive frames both criteria must be met

# --- Legacy (no longer used) ---
ARMED_TRIGGER_THRESHOLD: float = 5.0
PLATEAU_WINDOW: int = 15
PLATEAU_STD_THRESHOLD: float = 0.5
PLATEAU_RETURN_THRESHOLD: float = 3.0

SPARKLINE_MAX_SECONDS: int = 300    # max X-axis span (5 minutes); chart compresses to fit
ANALYSIS_FPS_TARGET: int = 30

# ---------------------------------------------------------------------------
# UI constants
# ---------------------------------------------------------------------------

INACTIVE_CARD_BORDER_COLOR = "#9ca3af"   # gray-400
INACTIVE_CARD_BG_COLOR = "#f9fafb"       # gray-50
CARD_BORDER_WIDTH_NORMAL = 2
CARD_BORDER_WIDTH_MIXING = 3
HEADER_BAR_HEIGHT = 28                   # px
SPARKLINE_HEIGHT = 100                   # px
METRICS_FONT_SIZE = 13                   # px, monospace
BASE_UNIT = 8                            # px
MARGIN = 12                              # px
CARD_GAP = 12                            # px
