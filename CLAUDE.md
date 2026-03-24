# Kineticolor — Computer Vision Mixing Analysis System

## Project Overview

A computer vision platform for real-time kinetic analysis of mixing phenomena in reaction vessels, inspired by the paper: *"Computer Vision for Kinetic Analysis of Lab- and Process-Scale Mixing Phenomena"* (Barrington et al., Org. Process Res. Dev. 2022, 26, 3073–3088).

**Primary use case:** Analyzing pH titration mixing in a 5L transparent bioreactor, replacing a pH probe that has 10–20 second delay, with real-time colorimetric feedback.

**Core idea:** The system does NOT understand chemistry. It tracks color change from a reference frame (frame 0) using 6 complementary metrics. "Mixing complete" = metrics plateau.

---

## Tech Stack

- **Language:** Python 3.11+
- **CV Library:** OpenCV (cv2)
- **GUI Framework:** PyQt6 (or PySide6)
- **Plotting:** pyqtgraph (real-time, embedded in GUI) + matplotlib (export)
- **Color Science:** scikit-image (rgb2lab) or colorspacious for RGB → CIE-L*a*b* conversion
- **Numerical:** NumPy, SciPy
- **Logging:** Python `logging` module with structured handlers
- **Config:** YAML for user-configurable parameters
- **Testing:** pytest

---

## Directory Structure

```
comp_viz_analysis/
├── CLAUDE.md                          # This file
├── config/
│   └── default_config.yaml            # All configurable parameters with defaults
├── src/
│   ├── __init__.py
│   ├── main.py                        # Application entry point
│   ├── core/                          # Core computation engine (NO GUI dependency)
│   │   ├── __init__.py
│   │   ├── video_reader.py            # Video file + live camera feed abstraction
│   │   ├── frame_processor.py         # ROI extraction, masking, color space conversion
│   │   ├── metrics/                   # Each metric is its own module
│   │   │   ├── __init__.py
│   │   │   ├── base_metric.py         # Abstract base class for all metrics
│   │   │   ├── delta_e.py             # ΔE metric (grand + spatially resolved)
│   │   │   ├── contact.py             # Contact (binary threshold + perimeter)
│   │   │   ├── glcm.py                # GLCM matrix builder (shared by contrast/homogeneity/energy)
│   │   │   ├── contrast.py            # Contrast from GLCM
│   │   │   ├── homogeneity.py         # Homogeneity from GLCM
│   │   │   ├── energy.py              # Energy / ASM from GLCM
│   │   │   └── variance.py            # Cell-based color variance
│   │   ├── grid_analyzer.py           # 5×5 (configurable) cell grid logic
│   │   ├── analysis_engine.py         # Orchestrates all metrics per frame, manages time series
│   │   └── export.py                  # CSV/Excel export of time-series data
│   ├── gui/                           # GUI layer (depends on core, never the reverse)
│   │   ├── __init__.py
│   │   ├── main_window.py             # Main application window layout
│   │   ├── video_panel.py             # Video display + ROI selection + exclusion mask drawing
│   │   ├── controls_panel.py          # Start/stop, config sliders, video upload, camera select
│   │   ├── plots_panel.py             # Real-time metric plots (pyqtgraph)
│   │   ├── roi_selector.py            # ROI rectangle drawing + exclusion mask painting
│   │   └── heatmap_overlay.py         # ΔE heatmap overlay on video frames
│   └── utils/
│       ├── __init__.py
│       ├── logger.py                  # Centralized logging setup
│       ├── color_convert.py           # RGB ↔ CIE-L*a*b* helpers
│       └── config_loader.py           # YAML config loader with validation
├── tests/
│   ├── __init__.py
│   ├── test_delta_e.py
│   ├── test_contact.py
│   ├── test_glcm.py
│   ├── test_contrast.py
│   ├── test_homogeneity.py
│   ├── test_energy.py
│   ├── test_variance.py
│   ├── test_grid_analyzer.py
│   ├── test_frame_processor.py
│   └── fixtures/                      # Test images / short video clips
│       └── README.md
├── docs/
│   └── reference_paper.pdf            # The source paper (symlink or copy)
└── requirements.txt
```

### Dependency Rule

**`core/` has ZERO dependency on `gui/`.** The core engine must be usable as a standalone library (headless analysis). The GUI imports from core, never the reverse.

---

## The 6 Mixing Metrics — Detailed Specification

### Reference Paper Equations

All metrics are computed within the user-selected **Region of Interest (ROI)**, with optional **exclusion mask** for obstructed areas.

---

### 1. ΔE (Delta E) — Color Distance from Reference

**What it measures:** Perceptually uniform color change from frame 0.

**Computation:**
1. Convert ROI pixels from RGB → CIE-L\*a\*b\* color space
2. For each pixel, compute Euclidean distance from the same pixel in frame 0:
   ```
   ΔE = √((L₂ - L₁)² + (a₂ - a₁)² + (b₂ - b₁)²)
   ```
3. **Grand (averaged) ΔE:** Mean of all pixel ΔE values across the ROI → single scalar per frame
4. **Spatially resolved ΔE:**
   - Per row average (N rows based on grid size)
   - Per column average (N columns based on grid size)
   - Per cell average (N×N grid)
   - Per-pixel ΔE heatmap for visualization

**Interpretation:**
- ΔE < 1: Not perceptible by human eye
- ΔE 1–2: Perceptible through close observation
- ΔE 2–10: Perceptible at a glance
- ΔE 11–49: Colors are more similar than opposite
- ΔE = 100: Exact opposite colors

**Output:** Time series of grand ΔE + spatially resolved arrays + heatmap frames

---

### 2. Contact — Binary Threshold Perimeter

**What it measures:** Amount of boundary between "light" and "dark" regions. Peaks during mixing transition.

**Computation:**
1. Convert ROI to grayscale
2. Apply binary threshold (user-configurable, default: 128):
   - Pixel ≥ threshold → white (1)
   - Pixel < threshold → black (0)
3. Calculate total perimeter of contact between white and black regions
   - Sum of all edges where a white pixel is adjacent to a black pixel (4-connectivity)

**Interpretation:**
- Low before mixing (uniform starting color)
- Rises during mixing (heterogeneous mix of reacted/unreacted zones)
- Falls after mixing (uniform final color)
- Local minima both before AND after mixing

**Output:** Time series of contact value (arbitrary units)

---

### 3. Contrast — from Gray-Level Co-Occurrence Matrix (GLCM)

**What it measures:** Magnitude of gray-level contrast for pixel pairs. High when image is heterogeneous.

**Computation:**
1. Convert ROI to grayscale
2. Quantize to N gray levels (configurable, default: 16)
3. Build GLCM: for each pixel, pair it with the pixel at offset (1 right, 1 down)
   - GLCM[i][j] counts how many times gray level i is paired with gray level j
4. Normalize GLCM:
   ```
   p_ij = a_ij / grand_sum(GLCM)     where Σ(p_ij) = 1
   ```
5. Compute contrast:
   ```
   contrast = Σᵢ Σⱼ |i - j|² · p_ij
   ```

**Interpretation:**
- High → visibly heterogeneous (mixing in progress)
- Low → visibly homogeneous (before or after mixing)
- Decreases toward 0 as mixing completes

**Output:** Time series of contrast value

---

### 4. Homogeneity — from GLCM

**What it measures:** How close the GLCM distribution is to the diagonal. Inverse of contrast conceptually.

**Computation:**
Uses the same normalized GLCM as contrast:
```
H = Σᵢ Σⱼ p_ij / (1 + |i - j|)
```

**Interpretation:**
- High → pixels are similar to their neighbors → uniform/homogeneous
- Low → pixels differ from neighbors → heterogeneous
- Increases as mixing completes

**Output:** Time series of homogeneity value (0 to 1 range)

---

### 5. Energy (Angular Second Moment / ASM) — from GLCM

**What it measures:** Amount of "block color" / textural uniformity.

**Computation:**
Uses the same normalized GLCM:
```
ASM = Σᵢ Σⱼ (p_ij)²
```

**Interpretation:**
- High → image has large uniform regions (single dominant color)
- Low → random noise or gradient (many different gray levels present)
- Reaches maximum when mixing is complete (uniform final color)
- A checkerboard or polka-dot pattern produces relatively HIGH ASM

**Output:** Time series of energy value (0 to 1 range)

---

### 6. Variance — Cell-Based Color Variance

**What it measures:** Spatial variation of average color across grid cells. Captures meso-mixing.

**Computation:**
1. Divide ROI into N×N grid (configurable, default: 5×5)
2. For each cell, compute the average color (per channel)
3. Compute variance of these cell averages across all cells
4. Computed for ALL color channels independently:
   - RGB: R, G, B
   - CIE-L\*a\*b\*: L\*, a\*, b\*
   - ΔE: variance of per-cell ΔE values

**Interpretation:**
- High variance → different parts of the vessel have different colors → inhomogeneous
- Low variance → uniform color across vessel → well-mixed
- Decreases as mixing completes

**Output:** Time series of variance per channel (7 channels: R, G, B, L*, a*, b*, ΔE)

---

## GLCM Shared Computation

**CRITICAL:** Contrast, Homogeneity, and Energy all use the SAME GLCM for a given frame. The GLCM is computed ONCE per frame and shared across all three metrics. The `glcm.py` module builds the matrix; the three metric modules consume it.

---

## Configurable Parameters (with defaults)

All stored in `config/default_config.yaml`. Changeable at runtime through GUI controls.

```yaml
# Frame processing
frame_skip: 1                    # Analyze every Nth frame (1 = every frame)
glcm_frame_skip: 1              # GLCM-specific frame skip (can be higher than frame_skip)

# Grid
grid_rows: 5                    # Number of rows in spatial grid
grid_cols: 5                    # Number of columns in spatial grid

# GLCM
glcm_gray_levels: 16            # Number of quantized gray levels (fewer = faster)
glcm_offset: [1, 1]             # Pixel pair offset [dx, dy] — (1 right, 1 down)

# Contact
contact_threshold: 128          # Grayscale threshold for binary contact analysis (0-255)

# Video / Camera
camera_index: 0                 # Default camera device index for live feed
video_fps_override: null        # Override video FPS if metadata is wrong

# Export
export_format: "csv"            # csv or xlsx
```

### Baseline Defaults (agreed)
- Grid: **5×5**
- GLCM gray levels: **16**
- Frame skip: **1** (every frame)
- Contact threshold: **128**
- GLCM offset: **(1, 1)** — one pixel right, one pixel down

All of these are changeable at runtime. The philosophy: **set a sensible default, let the user adjust, trial and test what works.**

---

## GUI Layout

### Main Window — Three-Panel Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Menu Bar: File | Settings | Export | Help                    │
├──────────────┬───────────────────────────────────────────────┤
│              │                                               │
│  VIDEO       │           PLOTS PANEL                         │
│  PANEL       │                                               │
│              │  ┌─────────────┐  ┌─────────────┐             │
│  [Live feed  │  │ Grand ΔE    │  │ Contrast    │             │
│   or video   │  │ vs Time     │  │ vs Time     │             │
│   playback]  │  └─────────────┘  └─────────────┘             │
│              │  ┌─────────────┐  ┌─────────────┐             │
│  ROI shown   │  │ Energy/ASM  │  │ Homogeneity │             │
│  as dashed   │  │ vs Time     │  │ vs Time     │             │
│  rectangle   │  └─────────────┘  └─────────────┘             │
│              │  ┌─────────────┐  ┌─────────────┐             │
│  ΔE heatmap  │  │ Contact     │  │ Variance    │             │
│  overlay     │  │ vs Time     │  │ vs Time     │             │
│  toggle      │  └─────────────┘  └─────────────┘             │
│              │                                               │
├──────────────┴───────────────────────────────────────────────┤
│  CONTROLS: [Upload Video] [Start Camera] [Select ROI]        │
│  [Start Analysis] [Stop] [Export Data]                        │
│  Grid: [5x5 ▾]  Frame Skip: [1 ▾]  GLCM Levels: [16 ▾]     │
│  Contact Threshold: [====128====]                             │
└──────────────────────────────────────────────────────────────┘
```

### GUI Features
1. **Video Panel:**
   - Displays uploaded video or live camera feed
   - ROI selection via click-drag rectangle
   - Exclusion mask painting (for obstructed areas) via brush tool
   - Toggle: raw frame vs ΔE heatmap overlay
   - Grid overlay toggle (shows the N×N cell divisions)

2. **Plots Panel:**
   - 6 real-time plots (one per metric), updating as frames are processed
   - Each plot: metric value (y-axis) vs time in seconds (x-axis)
   - Variance plot shows all 7 channels as separate colored lines
   - Spatially-resolved ΔE: switchable between grand, by-row, by-column views

3. **Controls Panel:**
   - Upload video file (mp4, avi, mov)
   - Select camera for live feed
   - Draw ROI / Draw exclusion mask
   - Start/Stop analysis
   - All configurable parameters adjustable via sliders/dropdowns
   - Export button (CSV with all metric time series)

---

## Edge Cases & Handling

### 1. Partially Obscured Bioreactor Bottom
- **Primary defense:** User draws ROI to include only the visible liquid region
- **Secondary:** Exclusion mask — user paints out regions WITHIN the ROI (e.g., clamps, labels, stand)
- **Grid cells >50% masked** are flagged invalid and excluded from variance calculations
- Masked pixels are excluded from ALL metrics (ΔE averaging, GLCM pairs, contact perimeter)

### 2. Reference Frame Selection
- Frame 0 of the video/recording is the default reference
- User can optionally select a different reference frame (e.g., if video starts mid-experiment)
- Reference frame is stored and used for all ΔE calculations

### 3. Camera Disconnection (Live Feed)
- If camera feed drops, pause analysis, show warning, attempt reconnect
- Do NOT reset reference frame on reconnect — maintain continuity

### 4. Very Long Videos
- Frame skip parameter handles this — user increases to reduce compute
- Data is appended to time series incrementally, not stored all in memory then processed
- Export can be done mid-analysis (exports data collected so far)

### 5. Lighting Changes
- The system does NOT correct for lighting. This is by design (matches the paper).
- The paper notes that lighting should be kept consistent during experiments.
- Log a warning if average brightness of the ROI changes drastically between consecutive frames.

---

## Logging Strategy

Using Python's `logging` module with the following levels:

| Level   | What gets logged                                                       |
|---------|------------------------------------------------------------------------|
| DEBUG   | Per-frame metric values, GLCM computation times, pixel counts          |
| INFO    | Analysis start/stop, video loaded, ROI selected, export completed      |
| WARNING | Masked cell >50%, camera reconnect, brightness shift detected          |
| ERROR   | Camera failure, file read error, invalid config values                 |

### Log destinations:
- **Console:** INFO and above (during development: DEBUG)
- **File:** `logs/kineticolor_YYYY-MM-DD_HHMMSS.log` — DEBUG and above
- **Rotate:** Max 10MB per file, keep 5 files

### What to log per frame (DEBUG level):
- Frame number and timestamp
- Time taken to compute each metric
- Total frame processing time
- Any masked cells skipped

---

## Performance Notes

- **GLCM is the bottleneck.** Quantizing to 16 gray levels (vs 256) reduces the matrix from 256×256 to 16×16 — massive speedup.
- **Frame skipping:** `frame_skip` applies to ALL metrics. `glcm_frame_skip` can be set independently higher if GLCM is too slow but you want ΔE at full frame rate. When `glcm_frame_skip > frame_skip`, GLCM metrics hold their last value for intermediate frames.
- **NumPy vectorization:** All pixel-level operations (ΔE, thresholding, GLCM building) MUST use NumPy array operations, not Python loops over pixels.
- **ROI extraction first:** Always crop to ROI before any computation — never process the full camera frame.

---

## Data Flow

```
Camera/Video → Frame → Crop to ROI → Apply Exclusion Mask
                                          │
                    ┌─────────────────────┼─────────────────────┐
                    │                     │                     │
                    ▼                     ▼                     ▼
              RGB → L*a*b*          RGB → Grayscale       RGB → Grayscale
                    │                     │                     │
                    ▼                     ▼                     ▼
              ΔE Metric            GLCM Builder          Contact Metric
              (grand + spatial)         │                (threshold + perimeter)
                    │              ┌────┼────┐
                    │              ▼    ▼    ▼
                    │         Contrast  H  Energy
                    │
                    ▼
              Grid Analyzer → Cell averages → Variance Metric
                                              (per channel: R,G,B,L*,a*,b*,ΔE)
                    │
                    ▼
              All metrics → Time Series Store → Real-time Plots + Export
```

---

## Coding Standards

- **Type hints** on all function signatures
- **Docstrings** on all public classes and methods (one-liner for simple, Google-style for complex)
- **No global state** — all configuration passed explicitly
- **Metrics follow a common interface** (base class) so new metrics can be added easily
- **Tests** for each metric using known synthetic images (e.g., solid color → ΔE should be 0; checkerboard → high contrast)
- **No premature optimization** — get it correct first, profile later
- **f-strings** for string formatting
- **pathlib.Path** for all file paths
