# Phase 1: Core Computation Engine — Design Spec

## Goal

Build the headless core engine for Kineticolor: video ingestion, frame processing, all 6 mixing metrics, grid analysis, and data export. No GUI dependency. Fully testable with synthetic images. Includes a CLI entry point for running analysis end-to-end.

## Scope

Everything under `src/core/` and `src/utils/`, plus `src/main.py`, `config/default_config.yaml`, `tests/`, and `requirements.txt`.

### In Scope
- CLI entry point (`src/main.py`) — accepts video path, ROI coordinates, config file, runs analysis, exports results
- Video file reading (mp4, avi, mov) via OpenCV, with `video_fps_override` support
- Frame processing: ROI extraction, exclusion mask application, color space conversion
- All 6 metrics: Delta E, Contact, Contrast, Homogeneity, Energy, Variance
- GLCM shared computation (built once per frame, consumed by 3 metrics)
- Independent `glcm_frame_skip` with hold-last-value behavior
- Grid analyzer (N×N cell decomposition)
- Analysis engine orchestrating metrics per frame with time series accumulation
- Data export (CSV and XLSX formats)
- YAML config loading with validation
- Centralized logging (console + rotating file, per CLAUDE.md spec)
- RGB <-> CIE-L*a*b* conversion utilities
- Brightness change warning detection
- Unit tests for every metric using synthetic images

### Out of Scope (Phase 2+)
- GUI (PyQt6)
- Live camera feed
- Real-time plotting
- ROI selection UI / exclusion mask painting UI
- Heatmap overlay rendering

## Architecture

```
config/default_config.yaml
src/
  __init__.py
  main.py                    # CLI entry point — argparse, wires core components
  core/
    __init__.py
    video_reader.py         # VideoReader class — wraps cv2.VideoCapture
    frame_processor.py      # FrameProcessor — ROI crop, mask, color conversion
    metrics/
      __init__.py
      base_metric.py        # Abstract base: compute(frame, ref_frame, mask) -> value
      delta_e.py            # DeltaEMetric
      contact.py            # ContactMetric
      glcm.py               # GLCMBuilder — builds normalized GLCM matrix
      contrast.py           # ContrastMetric (consumes GLCM)
      homogeneity.py        # HomogeneityMetric (consumes GLCM)
      energy.py             # EnergyMetric (consumes GLCM)
      variance.py           # VarianceMetric
    grid_analyzer.py        # GridAnalyzer — divides ROI into N×N cells
    analysis_engine.py      # AnalysisEngine — orchestrates per-frame pipeline
    export.py               # DataExporter — CSV and XLSX output
  utils/
    __init__.py
    logger.py               # setup_logger() with console + rotating file handlers
    color_convert.py        # rgb_to_lab(), lab_to_rgb() using skimage
    config_loader.py        # load_config() from YAML with defaults + validation
```

## Key Design Decisions

### 1. Metric Base Class

```python
class BaseMetric(ABC):
    @abstractmethod
    def compute(self, frame: np.ndarray, reference_frame: np.ndarray,
                mask: np.ndarray | None = None) -> dict[str, Any]:
        """Return a dict of named values for this frame."""
```

Each metric returns a dict so it can provide multiple outputs (e.g., Delta E returns `{"grand_delta_e": float, "row_averages": ndarray, "col_averages": ndarray, "cell_averages": ndarray}`).

### 2. GLCM Sharing and Independent Frame Skip

`GLCMBuilder` is not a metric itself. It's a utility that `AnalysisEngine` calls once per frame, then passes the resulting normalized GLCM matrix to `ContrastMetric`, `HomogeneityMetric`, and `EnergyMetric`. These three metrics accept the precomputed GLCM in their `compute()` method via an optional `glcm` parameter.

**`glcm_frame_skip` behavior:** When `glcm_frame_skip > frame_skip`, the GLCM is only recomputed every Nth analyzed frame. On intermediate frames, Contrast/Homogeneity/Energy hold their last computed value (the AnalysisEngine stores the previous GLCM results and reuses them). This allows Delta E, Contact, and Variance to run at full frame rate while GLCM metrics run at a lower rate for performance.

### 3. Frame Processing Pipeline

```
VideoReader.read_frame()
  -> FrameProcessor.crop_to_roi(frame, roi)
  -> FrameProcessor.apply_mask(cropped, mask)
  -> FrameProcessor.check_brightness(masked_roi, prev_brightness) — warn if drastic change
  -> AnalysisEngine processes the masked ROI:
       1. Convert to L*a*b* (for Delta E, Variance)
       2. Convert to grayscale (for GLCM, Contact)
       3. Compute Delta E first (needed by Variance for per-cell ΔE)
       4. Compute GridAnalyzer cell averages (needed by Variance)
       5. Build GLCM once (if this frame is a GLCM frame per glcm_frame_skip)
       6. Compute Contact, Contrast, Homogeneity, Energy, Variance
       7. Append results to time series
```

**Metric ordering constraint:** Delta E and GridAnalyzer must run before Variance, because Variance needs per-cell ΔE values. GLCM metrics and Contact have no ordering dependency.

### 4. VideoReader

Wraps `cv2.VideoCapture`. Provides:
- `open(path)` — open a video file
- `read_frame() -> tuple[bool, np.ndarray]` — read next frame
- `frame_count`, `fps`, `width`, `height` properties
- `fps` respects `video_fps_override` from config (uses override if set, otherwise video metadata)
- Handles frame skipping internally based on config
- Iterable interface for convenience

### 5. GridAnalyzer

Divides an ROI into N×N cells. Returns cell coordinates and handles:
- Computing per-cell average color (all channels)
- Flagging cells that are >50% masked as invalid
- Used by VarianceMetric and spatially-resolved Delta E

### 6. AnalysisEngine

The orchestrator. Holds:
- Reference frame (frame 0 by default)
- `set_reference_frame(frame_number: int)` — allows programmatic selection of a different reference frame (reads that frame from VideoReader, stores it)
- All metric instances
- Time series storage (list of dicts, one per analyzed frame)
- Last GLCM results (for hold-last-value on non-GLCM frames)
- Config reference

Drives the per-frame loop: read frame -> process -> compute metrics (respecting ordering and glcm_frame_skip) -> store results.

### 7. Mask Handling for Contact and GLCM

**Contact:** When computing perimeter, edges between a masked pixel and an unmasked pixel are NOT counted as contact edges. Only edges between two unmasked pixels (one white, one black after thresholding) count. The mask effectively shrinks the analysis region.

**GLCM:** Pixel pairs where either pixel is masked are skipped entirely. The normalization denominator is the count of valid (both unmasked) pairs only. This ensures the GLCM reflects only the visible region's texture.

### 8. Config

`config/default_config.yaml` with all defaults from CLAUDE.md:
```yaml
frame_skip: 1
glcm_frame_skip: 1
grid_rows: 5
grid_cols: 5
glcm_gray_levels: 16
glcm_offset: [1, 1]
contact_threshold: 128
camera_index: 0
video_fps_override: null
export_format: "csv"
```

`ConfigLoader` reads YAML, validates types/ranges, returns a typed dict or dataclass. Invalid values raise clear errors.

### 9. Export

`DataExporter` supports both CSV and XLSX formats (controlled by `export_format` config). One row per analyzed frame, columns for frame number, timestamp (seconds), and all metric outputs. Can be called mid-analysis (exports data collected so far). XLSX support uses `openpyxl`.

### 10. Logging

`setup_logger()` configures:
- **Console handler:** INFO and above
- **File handler:** DEBUG and above, to `logs/kineticolor_YYYY-MM-DD_HHMMSS.log`
- **Rotation:** `RotatingFileHandler`, max 10MB per file, keep 5 backup files

Per-frame DEBUG logging includes: frame number, timestamp, per-metric computation time, total frame processing time, any masked cells skipped.

### 11. Brightness Change Warning

`FrameProcessor` tracks average brightness of the unmasked ROI. If brightness changes by more than a configurable threshold (default: 20%) between consecutive frames, log a WARNING. This does not halt analysis — it's informational only, per the paper's note that lighting should be consistent.

### 12. CLI Entry Point (`src/main.py`)

```
python -m src.main --video path/to/video.mp4 \
                   --roi x,y,w,h \
                   --config config/default_config.yaml \
                   --output results.csv \
                   --reference-frame 0
```

All arguments have sensible defaults (ROI defaults to full frame, config defaults to `config/default_config.yaml`, reference frame defaults to 0). The CLI wires together VideoReader, FrameProcessor, AnalysisEngine, and DataExporter.

## Metric Specifications

All formulas and interpretations are defined in CLAUDE.md sections "The 6 Mixing Metrics." The implementation follows those exactly:

| Metric | Input | Output Keys |
|--------|-------|-------------|
| Delta E | L*a*b* frame + ref | `grand_delta_e`, `pixel_delta_e` (heatmap), `row_avg`, `col_avg`, `cell_avg` |
| Contact | Grayscale frame + mask | `contact_perimeter` |
| Contrast | Normalized GLCM | `contrast` |
| Homogeneity | Normalized GLCM | `homogeneity` |
| Energy | Normalized GLCM | `energy` |
| Variance | RGB + L*a*b* + cell Delta E | `variance_r`, `variance_g`, `variance_b`, `variance_l`, `variance_a`, `variance_b_star`, `variance_delta_e` |

## Testing Strategy

Each metric gets a test file with synthetic images:
- **Delta E:** Solid color vs itself -> 0. Solid color vs different solid -> known value.
- **Contact:** Solid image -> 0. Half black/half white -> perimeter = boundary length. With mask -> masked boundary excluded.
- **Contrast:** Uniform gray -> 0. Checkerboard -> high value.
- **Homogeneity:** Uniform -> 1.0. Random noise -> low value.
- **Energy:** Uniform -> 1.0 (single GLCM entry). Random -> low value.
- **Variance:** Uniform color across cells -> 0. Different colors per cell -> high.
- **GridAnalyzer:** Known image, verify cell coordinates and averages. Verify >50% masked cells flagged.
- **FrameProcessor:** ROI crop produces correct dimensions; mask zeros out correct pixels. Brightness warning triggers on large shift.
- **GLCM with mask:** Verify masked pixel pairs are excluded from GLCM.

## Dependencies

```
numpy
opencv-python
scikit-image
scipy
pyyaml
openpyxl
pytest
```

## Performance Considerations

- All pixel operations use NumPy vectorization (no Python loops over pixels)
- GLCM quantized to 16 gray levels (16x16 matrix, not 256x256)
- ROI cropped before any computation
- Frame skipping configurable (both general and GLCM-specific)
- Time series data appended incrementally (no full-video buffering)
