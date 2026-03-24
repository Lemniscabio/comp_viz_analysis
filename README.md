# Kineticolor

A computer vision desktop application for real-time kinetic analysis of mixing phenomena in reaction vessels.

Built for analyzing pH titration mixing in transparent bioreactors — replacing pH probes that have 10-20 second delay with real-time colorimetric feedback from video.

**How it works:** The system tracks color change from a reference frame (frame 0) using 6 complementary metrics. When all metrics plateau, mixing is complete.

Based on: *"Computer Vision for Kinetic Analysis of Lab- and Process-Scale Mixing Phenomena"* (Barrington et al., Org. Process Res. Dev. 2022, 26, 3073-3088).

---

## Installation

```bash
# Clone the repository
git clone https://github.com/Lemniscabio/comp_viz_analysis.git
cd comp_viz_analysis

# Create a conda environment (recommended)
conda create -n kineticolor python=3.12
conda activate kineticolor

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.10+ (3.12 recommended)
- OpenCV, NumPy, SciPy, scikit-image
- PyQt6, pyqtgraph (for GUI)
- PyYAML, openpyxl (for config and export)

---

## Two Ways to Use

### 1. Desktop GUI (recommended)

```bash
conda activate kineticolor
python -m src.main
```

This opens the full graphical application. Workflow:

1. **Open Video** — load an MP4, AVI, or MOV file
2. **Select ROI** (optional) — draw a rectangle around the region of interest (e.g., the liquid in the bioreactor). If skipped, the full frame is analyzed.
3. **Draw Mask** (optional) — paint over areas to exclude from analysis (e.g., tubes, clamps, labels blocking the view)
4. **Start Analysis** — runs all 6 metrics on every frame
5. **Export Data** — save results to CSV or Excel

#### GUI Controls

| Control | What it does |
|---------|-------------|
| **Select ROI** | Draw a rectangle around the area to analyze. Drag inside to move, drag corners to resize. Click button again to clear. |
| **Draw Mask** | Click and drag to paint exclusion areas (shown in red). Scroll wheel changes brush size. |
| **Erase Mask** | Click and drag to restore previously masked areas. |
| **Grid** | Toggle grid overlay showing the NxN cell divisions on the video. |
| **Heatmap** | Toggle Delta-E heatmap overlay. Blue = little change from frame 0. Red = large color change. |
| **Toggle Video** | Show/hide the video panel to give more space to plots. |

#### Configuration Controls

| Setting | Default | What it changes |
|---------|---------|----------------|
| **Grid** | 5x5 | Number of cells for spatial analysis. More cells = finer resolution, fewer = faster. |
| **Frame Skip** | 1 | Analyze every Nth frame. 1 = every frame. Higher = faster but less temporal detail. |
| **GLCM Skip** | 1 | Recompute texture metrics (Contrast, Homogeneity, Energy) every Nth analyzed frame. These are the slowest metrics. Values hold steady between updates. |
| **GLCM Levels** | 16 | Gray level quantization for texture analysis. 16 is the sweet spot (fast + robust). |
| **Threshold** | 128 | Grayscale cutoff for the Contact metric (0-255). Pixels above = white, below = black. Contact counts the boundary between them. Adjust based on your liquid color. |
| **Export** | csv | File format for data export (CSV or Excel). |

### 2. Command Line (headless)

```bash
conda activate kineticolor
python -m src.main --video path/to/video.mp4
```

Runs the analysis without any GUI. Results are printed as a progress bar and exported to a file.

#### CLI Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | (required) | Path to the video file to analyze. |
| `--roi` | full frame | Region of interest as `x,y,w,h`. Example: `--roi 50,100,400,300` means start at 50px from left edge, 100px from top edge, grab a 400px wide by 300px tall rectangle. |
| `--config` | `config/default_config.yaml` | Path to a YAML config file. |
| `--output` | `results.csv` | Output file path. Use `.xlsx` extension for Excel format. |
| `--reference-frame` | `0` | Which frame to use as the baseline for Delta-E. Default is the first frame. |

#### Examples

```bash
# Analyze with default settings, export to CSV
python -m src.main --video experiment.mp4

# Analyze a specific region, export to Excel
python -m src.main --video experiment.mp4 --roi 50,100,400,300 --output results.xlsx

# Use custom config and different reference frame
python -m src.main --video experiment.mp4 --config my_config.yaml --reference-frame 30

# Skip frames for faster processing
# (edit config/default_config.yaml and set frame_skip: 5)
python -m src.main --video long_experiment.mp4
```

---

## The 6 Mixing Metrics

The system does NOT understand chemistry. It tracks visual changes using these 6 complementary metrics:

### 1. Delta-E (Color Distance)
Measures perceptual color change from the reference frame, per pixel, in CIE-L\*a\*b\* color space. The single most important metric. When Delta-E plateaus, the color has stabilized.

### 2. Contact (Binary Perimeter)
Counts the boundary between "light" and "dark" regions after thresholding. Rises during mixing (heterogeneous zones appear), falls when mixing completes (uniform color).

### 3. Contrast (from GLCM)
How much gray-level variation exists between neighboring pixels. High during active mixing, drops to near-zero when homogeneous.

### 4. Homogeneity (from GLCM)
Inverse of contrast conceptually. High = uniform/well-mixed. Low = heterogeneous/mixing in progress.

### 5. Energy / ASM (from GLCM)
Amount of "block color" uniformity. Reaches maximum when mixing is complete and the vessel has one dominant color.

### 6. Variance (Cell-based)
Spatial variation of average color across grid cells, computed for all 7 channels (R, G, B, L\*, a\*, b\*, Delta-E). High = different parts of the vessel have different colors. Drops as mixing completes.

**Note:** Contrast, Homogeneity, and Energy all share the same GLCM (Gray-Level Co-occurrence Matrix), which is computed once per frame for efficiency.

---

## Configuration

All parameters are in `config/default_config.yaml`:

```yaml
frame_skip: 1              # Analyze every Nth frame
glcm_frame_skip: 1         # GLCM-specific frame skip
grid_rows: 5               # Grid rows for spatial analysis
grid_cols: 5               # Grid columns
glcm_gray_levels: 16       # Gray level quantization (16 recommended)
glcm_offset: [1, 1]        # Pixel pair offset for GLCM
contact_threshold: 128     # Binary threshold for Contact metric
export_format: "csv"       # csv or xlsx
```

These can also be changed at runtime through the GUI controls.

---

## Output Data

The exported CSV/Excel file contains one row per analyzed frame with these columns:

| Column | Description |
|--------|-------------|
| `frame_number` | Zero-based frame index |
| `timestamp` | Time in seconds |
| `grand_delta_e` | Average color distance from reference frame |
| `contact_perimeter` | Boundary length between light/dark regions |
| `contrast` | Texture contrast from GLCM |
| `homogeneity` | Texture homogeneity from GLCM |
| `energy` | Texture energy/ASM from GLCM |
| `variance_r`, `_g`, `_b` | Spatial variance per RGB channel |
| `variance_l`, `_a`, `_b_star` | Spatial variance per L\*a\*b\* channel |
| `variance_delta_e` | Spatial variance of per-cell Delta-E |

---

## Project Structure

```
comp_viz_analysis/
  src/
    main.py                    # Entry point (GUI or CLI)
    core/                      # Computation engine (no GUI dependency)
      video_reader.py          # Video file reading
      frame_processor.py       # ROI cropping, masking, color conversion
      grid_analyzer.py         # NxN cell grid decomposition
      analysis_engine.py       # Orchestrates all metrics per frame
      export.py                # CSV/Excel export
      metrics/
        delta_e.py             # Delta-E color distance
        contact.py             # Binary threshold perimeter
        glcm.py                # GLCM matrix builder (shared)
        contrast.py            # Contrast from GLCM
        homogeneity.py         # Homogeneity from GLCM
        energy.py              # Energy/ASM from GLCM
        variance.py            # Cell-based color variance
    gui/                       # Desktop GUI (depends on core, never reverse)
      main_window.py           # Main application window
      video_panel.py           # Video display with overlays
      controls_panel.py        # Buttons, sliders, dropdowns
      plots_panel.py           # 6 real-time metric charts
      roi_selector.py          # ROI rectangle + mask brush tools
      heatmap_overlay.py       # Delta-E heatmap visualization
      analysis_worker.py       # Background thread for analysis
    utils/
      logger.py                # Centralized logging
      color_convert.py         # RGB <-> CIE-L*a*b* conversion
      config_loader.py         # YAML config with validation
  config/
    default_config.yaml        # Default parameters
  tests/                       # Unit + integration tests
```

**Architecture rule:** `core/` has zero dependency on `gui/`. The core engine works as a standalone library for headless analysis.

---

## Running Tests

```bash
conda activate kineticolor
pytest tests/ -v
```

60 tests covering all metrics, frame processing, grid analysis, config loading, and end-to-end integration.

---

## Cross-Platform

Built with PyQt6 (Qt framework). Runs on macOS, Windows, and Linux without code changes.

---

## License

[Add your license here]
