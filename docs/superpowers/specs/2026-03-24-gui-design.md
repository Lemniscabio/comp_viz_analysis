# Phase 2: GUI — Design Spec

## Goal

Build a PyQt6 GUI for Kineticolor with three-panel layout: video display (left), real-time metric plots (right), controls (bottom). Supports video file upload, live camera feed, ROI selection, exclusion mask painting, heatmap overlay, and data export.

## Scope

Everything under `src/gui/`, plus updates to `src/main.py` for GUI launch mode.

### In Scope
- Main window with three-panel layout and menu bar
- Video panel: frame display, ROI rectangle drawing, exclusion mask brush, grid overlay (with invalid cell indicators), ΔE heatmap overlay toggle
- Plots panel: 6 real-time pyqtgraph plots (ΔE with view switching, Contrast, Homogeneity, Energy, Contact, Variance)
- Controls panel: upload video, start camera, select ROI, draw exclusion mask, start/stop analysis, config sliders/dropdowns, export button
- Background thread for analysis (QThread + signals)
- Camera reconnection handling
- Reference frame selection
- Brightness change warning surfaced in status bar
- Mid-analysis export support

### Out of Scope
- Video recording/saving
- Multi-video batch processing
- Remote camera feeds

## Architecture

```
src/gui/
  __init__.py
  main_window.py        # QMainWindow — three-panel layout, menu bar, orchestration
  video_panel.py        # QWidget — video display, overlay rendering, mouse interaction
  controls_panel.py     # QWidget — buttons, sliders, dropdowns, file dialog
  plots_panel.py        # QWidget — 6 pyqtgraph PlotWidgets in 3×2 grid
  roi_selector.py       # ROI rectangle + exclusion mask brush logic
  heatmap_overlay.py    # Renders per-pixel ΔE as color-mapped overlay
  analysis_worker.py    # QThread — runs AnalysisEngine in background (new file, not in CLAUDE.md structure but necessary for GUI threading)
```

## Key Design Decisions

### 1. Threading Model

`AnalysisWorker(QThread)` runs the frame loop in a background thread. Communicates with the GUI via Qt signals only — no shared mutable state.

Signals emitted by AnalysisWorker:
- `frame_ready(int, np.ndarray, dict)` — frame number, rendered frame (with overlays), metrics dict
- `progress(int, int)` — current frame, total frames
- `analysis_finished()` — video ended or stopped
- `error_occurred(str)` — exception message
- `brightness_warning(str)` — surfaced from logger when brightness shift detected

Slots on MainWindow:
- `on_frame_ready()` — updates video panel + plots panel
- `on_progress()` — updates progress info in controls
- `on_analysis_finished()` — re-enables controls
- `on_error()` — shows error dialog
- `on_brightness_warning()` — shows warning in status bar

### 2. Video Panel

Displays frames as QPixmap on a QLabel. Supports three interaction modes:
- **View mode** (default): just displays frames
- **ROI mode**: click-drag draws a rubber band rectangle, stores (x,y,w,h)
- **Mask mode**: brush tool paints exclusion areas on the frame, stored as binary mask

Overlay toggles (composited before display):
- Grid overlay: draws N×N cell divisions on the frame. Cells that are >50% masked are drawn with a red/hatched indicator to show they're flagged invalid.
- ΔE heatmap: color-mapped per-pixel ΔE blended onto the frame

### 3. Plots Panel

6 `pyqtgraph.PlotWidget` instances in a 3×2 grid:
```
[ ΔE          ] [ Contrast    ]
[ Energy/ASM  ] [ Homogeneity ]
[ Contact     ] [ Variance    ]
```

Each plot: metric value (y) vs time in seconds (x). Auto-scrolling. Variance plot shows 7 colored lines (R, G, B, L*, a*, b*, ΔE).

**ΔE view switching:** The ΔE plot has a dropdown/toggle to switch between:
- Grand ΔE (single line — default)
- By-row (N lines, one per grid row)
- By-column (N lines, one per grid column)

Data is appended incrementally via `plot.setData()` on each `frame_ready` signal.

### 4. Controls Panel

Layout (horizontal bar at bottom):
```
[Upload Video] [Start Camera] [Select ROI] [Draw Mask] | [Start] [Stop] [Export]
Grid: [5×5 ▾]  Frame Skip: [1 ▾]  GLCM Skip: [1 ▾]  GLCM Levels: [16 ▾]  Threshold: [===128===]
Export Format: [csv ▾]  FPS Override: [auto ▾]
```

- File dialog for video upload (mp4, avi, mov)
- Camera index selector (dropdown of available cameras)
- ROI and mask buttons toggle the video panel interaction mode
- Start/Stop enable/disable based on state (video loaded, ROI selected)
- Config controls: QComboBox for grid size, frame_skip, glcm_frame_skip, glcm_gray_levels, export_format, video_fps_override; QSlider for contact_threshold
- Export triggers DataExporter with current results — **works during analysis** (exports data collected so far) and when stopped
- Reference frame: right-click on video panel or menu option to set current frame as reference

### 5. ROI Selector

Two tools on the video panel:
- **Rectangle tool**: rubber band selection, shows dashed rectangle overlay. Stores `(x, y, w, h)` in frame coordinates (accounting for display scaling).
- **Brush tool**: circular brush, user paints areas to exclude. Stores a binary mask `(H, W)` where 1=keep, 0=exclude. Brush size adjustable via scroll wheel.

Coordinate mapping: video frames are scaled to fit the panel. Mouse coordinates are mapped back to original frame coordinates for ROI/mask storage.

### 6. Heatmap Overlay

Takes `pixel_delta_e` array from DeltaEMetric, applies a colormap (e.g., `cv2.COLORMAP_JET`), blends with the original frame at configurable opacity (default 50%). Toggled via a checkbox in controls.

### 7. Camera Support

Uses VideoReader with `camera_index` instead of file path. Additional handling:
- If camera feed drops, pause analysis, show warning banner, attempt reconnect every 2 seconds
- Do NOT reset reference frame on reconnect
- "Start Camera" button shows available camera indices

### 8. Menu Bar

```
File: Open Video | Quit
Settings: Configure Parameters (opens config dialog)
Export: Export Data (CSV/XLSX)
Help: About
```

### 9. State Machine

```
IDLE → video loaded → READY → ROI selected → CONFIGURED → Start → RUNNING → Stop → CONFIGURED
                                                          → video ends → CONFIGURED
RUNNING → camera disconnect → PAUSED → reconnect → RUNNING
```

Controls enable/disable based on state:
- IDLE: only Upload/Camera enabled
- READY: Upload/Camera/ROI enabled
- CONFIGURED: all enabled
- RUNNING: only Stop/Export enabled
- PAUSED: only Stop/Export enabled

### 10. Launch Mode

Update `src/main.py` to support both CLI and GUI:
```bash
# GUI mode (default when no --video flag)
python -m src.main

# CLI mode (when --video is provided)
python -m src.main --video path/to/video.mp4
```

## Dependencies (additions to requirements.txt)

```
PyQt6>=6.5
pyqtgraph>=0.13
```

## Performance Considerations

- Video frames scaled to panel size before display (not full resolution)
- Plots use pyqtgraph for hardware-accelerated rendering
- Heatmap overlay computed only when toggled on
- Frame processing in QThread keeps UI responsive
- Signal/slot mechanism prevents thread-safety issues
