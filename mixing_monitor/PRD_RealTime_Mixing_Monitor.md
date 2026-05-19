# PRD: Real-Time Multi-Vessel Mixing Monitor

## Product Overview

A Windows desktop application for real-time mixing-time detection across 1–4 bioreactor vessels. The system captures live video from a wired iPhone (or USB webcam), crops the feed into per-vessel regions, and runs colorimetric analysis on each crop in parallel. When the liquid in a vessel transitions from pink to transparent (pH indicator mixing), the system detects the plateau and reports the mixing time.

**This application is an extension of the existing Kineticolor project** (`comp_viz_analysis/`). It lives in a **fully self-contained** `mixing_monitor/` folder at the project root. It does NOT modify, import from, or depend on anything in `src/`, `config/`, `scripts/`, or `tests/`. All code, config, and output for this system are contained within `comp_viz_analysis/mixing_monitor/`.

---

## Critical Constraint: Offline Operation

The user has **no Wi-Fi access** in the lab during mixing studies. Every design decision must respect this:

- **No network dependencies at runtime.** No web APIs, no cloud services, no telemetry, no package downloads during use.
- **All Python dependencies must be pre-installed** via `pip install -r mixing_monitor/requirements.txt` before going to the lab. The requirements file must list every dependency explicitly with pinned versions.
- **Camera input must be wired.** iPhone connected via USB cable through a virtual webcam driver (Camo by Reincubate, or EpocCam). No Wi-Fi-based streaming (IP Webcam, AirPlay, etc.).
- **No license-check or activation flows** that require internet in any dependency.
- **All fonts, icons, and assets must be bundled** or use system defaults. No CDN loads, no Google Fonts.

---

## System Architecture

### Two Separate Applications

The system is split into two standalone PyQt6 apps that communicate through a shared JSON config file. This keeps each app simple and single-purpose.

```
┌─────────────────────┐         vessel_rois.json         ┌──────────────────────┐
│  Calibration App     │ ──── writes ──────────────────▶ │  Monitor App          │
│  (run once per       │                                  │  (run during          │
│   camera position)   │                                  │   experiment)         │
└─────────────────────┘                                  └──────────────────────┘
```

### Project Structure

All new code lives in `mixing_monitor/` at the project root. **Nothing outside this folder is created, modified, or deleted.** The existing `src/`, `config/`, `scripts/`, and `tests/` directories are completely untouched.

```
comp_viz_analysis/
  src/                              # EXISTING — DO NOT MODIFY
  config/                           # EXISTING — DO NOT MODIFY
  scripts/                          # EXISTING — DO NOT MODIFY
  tests/                            # EXISTING — DO NOT MODIFY
  mixing_monitor/                   # NEW — entire real-time system lives here
    __init__.py
    requirements.txt                # Additional deps for this subsystem only
    README.md                       # Setup and usage instructions
    common/
      __init__.py
      camera.py                     # CameraSource class + device enumeration
      roi_config.py                 # Load/save vessel_rois.json
      vessel_analyzer.py            # Per-vessel metric computation + plateau detection
      constants.py                  # Shared constants (colors, limits, defaults)
    calibration/
      __init__.py
      calibration_app.py            # PyQt6 app entry point
      calibration_window.py         # Main window layout and logic
      roi_canvas.py                 # Interactive rectangle drawing widget on video
    monitor/
      __init__.py
      monitor_app.py                # PyQt6 app entry point
      monitor_window.py             # Main window layout and logic
      capture_thread.py             # QThread for frame grabbing
      analysis_pool.py              # ThreadPoolExecutor coordinator
      vessel_card.py                # Single vessel display widget (video + metrics)
      results_bar.py                # Bottom status bar widget
      session_logger.py             # CSV log writer per session
    config/
      vessel_rois.json              # Written by calibration, read by monitor
    results/                        # Session logs written here by monitor app
```

**Isolation rule:** `mixing_monitor/` has zero imports from `src/`. It reimplements the specific Lab conversion and Delta-E logic it needs (which is ~10 lines of OpenCV/NumPy) rather than coupling to `src/core/`. This means Kineticolor can evolve independently without breaking the real-time monitor, and vice versa.

### Entry Points

```bash
# Calibration (run once per camera setup)
python -m mixing_monitor.calibration.calibration_app

# Live Monitor (run during experiments)
python -m mixing_monitor.monitor.monitor_app
```

---

## Tech Stack

| Component | Choice | Reason |
|-----------|--------|--------|
| Language | Python 3.12 | Matches existing Kineticolor |
| UI Framework | PyQt6 | Already a project dependency; native Windows feel |
| Video Capture | OpenCV (`cv2.VideoCapture`) | Standard, works with virtual webcam drivers |
| Image Processing | OpenCV + NumPy | GIL-releasing C extensions for true thread parallelism |
| Live Plots | pyqtgraph | Already a project dependency; fast real-time plotting |
| Parallel Analysis | `concurrent.futures.ThreadPoolExecutor` | See threading rationale below |
| Config Format | JSON | Human-readable, no external deps |
| Session Logging | CSV (via Python stdlib) | Simple, offline, importable into Kineticolor post-hoc |

### Why Threading (Not Multiprocessing)

- The heavy work (OpenCV color conversion, NumPy mean/std) runs in C extensions that **release the GIL**. True parallelism is achieved without multiprocessing.
- Avoids serializing ~200×300 pixel frames through IPC queues. Serialization overhead can exceed the analysis cost itself.
- All three analyzers share reference frames and config in the same memory space. No duplication.
- Simpler error handling — no zombie processes on crash.

### New Dependencies (`mixing_monitor/requirements.txt`)

This file lists ALL dependencies needed for the mixing monitor (including shared ones like OpenCV and PyQt6). It is a standalone file — the user runs `pip install -r mixing_monitor/requirements.txt` independently of the main project's requirements.

```
opencv-python>=4.8.0
numpy>=1.24.0
PyQt6>=6.5.0
pyqtgraph>=0.13.3
pygrabber>=0.2          # Windows DirectShow device enumeration (for camera name discovery)
```

If `pygrabber` proves problematic, fall back to index-only enumeration with OpenCV (try indices 0–9, show which ones open successfully). This fallback must be implemented.

---

## Camera Source (`common/camera.py`)

### Device Enumeration

```python
class CameraSource:
    @staticmethod
    def list_devices() -> list[dict]:
        """Return list of available video devices.

        Each dict: {"index": int, "name": str}

        Uses pygrabber on Windows for named devices.
        Falls back to probing indices 0-9 via cv2.VideoCapture if pygrabber unavailable.
        """

    def open(self, device_index: int, resolution: tuple[int, int] = (1280, 720)) -> bool:
        """Open the device. Returns True on success."""

    def read(self) -> tuple[bool, np.ndarray | None]:
        """Read a single frame. Returns (success, frame)."""

    def release(self) -> None:
        """Release the device."""

    @property
    def is_opened(self) -> bool: ...

    @property
    def actual_resolution(self) -> tuple[int, int]: ...

    @property
    def actual_fps(self) -> float: ...
```

### Resolution

Request **1280×720 (720p)** from the device. If the device cannot provide 720p, accept whatever it gives and log a warning. Store the actual resolution in the config so the monitor app knows what to expect.

### Frame Format

All frames are BGR `np.ndarray` as returned by OpenCV. Color conversion to Lab happens in the analyzer, not in the camera layer.

---

## Config File: `vessel_rois.json`

Written by the calibration app. Read by the monitor app. Lives in `mixing_monitor/config/` by default, but both apps accept a `--config` flag to override.

### Schema

```json
{
  "version": 1,
  "camera": {
    "device_index": 1,
    "device_name": "Camo (iPhone)",
    "resolution": [1280, 720]
  },
  "vessel_count": 3,
  "vessels": [
    {
      "id": 1,
      "label": "V1",
      "color": "#22c55e",
      "roi": [120, 180, 280, 400]
    },
    {
      "id": 2,
      "label": "V2",
      "color": "#3b82f6",
      "roi": [410, 175, 280, 405]
    },
    {
      "id": 3,
      "label": "V3",
      "color": "#eab308",
      "roi": [700, 170, 285, 410]
    }
  ]
}
```

### Field Definitions

| Field | Type | Description |
|-------|------|-------------|
| `version` | int | Schema version. Always `1` for now. |
| `camera.device_index` | int | OpenCV device index selected during calibration. |
| `camera.device_name` | str | Human-readable device name (for display only). |
| `camera.resolution` | [int, int] | [width, height] that was active during calibration. |
| `vessel_count` | int | Number of active vessels (1–4). |
| `vessels` | array | One entry per active vessel, ordered by ID. |
| `vessels[].id` | int | Vessel number (1–4). |
| `vessels[].label` | str | Display label. Default: "V1", "V2", "V3", "V4". User-editable in calibration. |
| `vessels[].color` | str | Hex color for UI border and overlays. Fixed per ID. |
| `vessels[].roi` | [int, int, int, int] | `[x, y, width, height]` in pixels, relative to the camera resolution. |

### Vessel Color Mapping (fixed)

| Vessel ID | Color Name | Hex | Rationale |
|-----------|-----------|------|-----------|
| 1 | Green | `#22c55e` | Matches green cap on leftmost vessel |
| 2 | Blue | `#3b82f6` | Matches blue cap on center vessel |
| 3 | Yellow | `#eab308` | Matches yellow cap on right vessel |
| 4 | Red | `#ef4444` | Matches red clips / 4th vessel if present |

---

## App 1: Calibration Tool

### Purpose

One-time setup to select the camera device and draw bounding boxes around 1–4 vessels. Saves `vessel_rois.json`.

### Launch

```bash
python -m mixing_monitor.calibration.calibration_app [--config path/to/vessel_rois.json]
```

### Window Specifications

- **Title:** "Vessel Calibration — Kineticolor"
- **Default size:** 1100 × 750 px
- **Resizable:** Yes
- **Minimum size:** 900 × 600 px

### Layout

```
┌──────────────────────────────────────────────────────────────┐
│  Vessel Calibration — Kineticolor                    [─][□][×]│
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  Camera: [ (select device)          ▾ ]   [ 🔄 Refresh ]    │
│  Status: ● Connected — 1280×720 @ 30fps                     │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│                                                              │
│  ┌────────────────────────────────────────────────────────┐  │
│  │                                                        │  │
│  │                   LIVE CAMERA FEED                     │  │
│  │                                                        │  │
│  │          ┌──── V1 ────┐    ┌──── V2 ────┐             │  │
│  │          │ (green     │    │ (blue      │             │  │
│  │          │  border)   │    │  border)   │             │  │
│  │          └────────────┘    └────────────┘             │  │
│  │                  ┌──── V3 ────┐                       │  │
│  │                  │ (yellow    │                       │  │
│  │                  │  border)   │                       │  │
│  │                  └────────────┘                       │  │
│  │                                                        │  │
│  └────────────────────────────────────────────────────────┘  │
│                                                              │
│  Number of vessels:  (1) (2) (3) (4)    ← radio buttons     │
│                                                              │
│  Instructions: Click and drag on the feed to draw a          │
│  bounding box for each vessel. Drag inside to move,          │
│  drag corners/edges to resize. Right-click to delete.        │
│                                                              │
├──────────────────────────────────────────────────────────────┤
│  [ Save Config ]    [ Load Existing Config ]                 │
│  Status: Saved to mixing_monitor/config/vessel_rois.json ✓                  │
└──────────────────────────────────────────────────────────────┘
```

### User Flow

1. **App opens.** Camera dropdown is populated with available devices. If only one device exists, it is auto-selected and the feed starts immediately.
2. **User selects a camera** from the dropdown. The live feed starts in the canvas area. If the device fails to open, show an inline error: "Cannot open device. Check connection and try Refresh."
3. **User selects number of vessels** via radio buttons (1–4). Default: 3. Changing this number clears any existing rectangles on the canvas.
4. **User draws bounding boxes** by clicking and dragging on the live feed. Each new rectangle is assigned the next available vessel ID and color. Drawing is only allowed up to the selected vessel count.
5. **User adjusts boxes** by dragging to reposition or dragging corners/edges to resize. Visual handles (small squares) appear on corners and edge midpoints when a box is hovered or selected.
6. **User right-clicks a box** to delete it. The remaining boxes are renumbered (V1, V2, V3 order is always left-to-right by x-coordinate of the box center).
7. **User clicks "Save Config."** The JSON is written. A status message confirms the save path.
8. **User can click "Load Existing Config"** to import a previously saved JSON. The camera device, vessel count, and ROI boxes are restored. If the camera device in the config is not currently available, show a warning and let the user select a different device. The ROIs still load onto the feed.

### ROI Canvas Behavior

- The live feed is scaled to fit the canvas widget while maintaining aspect ratio. Black bars fill any remaining space (letterboxing).
- All ROI coordinates are stored in **original camera pixel coordinates**, not scaled canvas coordinates. The canvas handles the mapping internally.
- Rectangles are drawn with a 2px solid border in the vessel's assigned color.
- The vessel label ("V1", "V2", etc.) is drawn in a small filled rectangle (pill shape) at the top-left corner of each ROI box.
- Minimum ROI size: 50×50 pixels (in camera coordinates). Attempts to draw smaller are ignored.
- ROIs **may overlap.** This is valid (if vessels are close together in the frame, some shared pixels are acceptable).
- ROIs **must not extend outside the frame.** Clamp to frame boundaries during drawing and resizing.

### Edge Cases — Calibration

| Scenario | Behavior |
|----------|----------|
| No camera devices found | Dropdown shows "(No cameras found)". Feed area shows a gray placeholder with text: "No camera detected. Connect a camera and click Refresh." |
| Camera disconnects during calibration | Feed freezes. Status changes to "● Disconnected". A yellow banner appears: "Camera disconnected. Reconnect and click Refresh." Drawn ROIs persist. |
| User clicks Save with fewer boxes than vessel count | Save button is disabled (grayed out) until the number of drawn boxes equals the selected vessel count. Tooltip: "Draw N bounding boxes to save." |
| User loads a config for a different resolution | ROIs are scaled proportionally to the current camera resolution. A warning is shown: "Config was saved at 1920×1080, current camera is 1280×720. ROIs have been scaled. Verify positions." |
| Config file is corrupted or invalid | Show error dialog: "Cannot load config: [reason]. Start fresh or select a different file." |
| User closes without saving | If ROIs have been drawn/modified, show confirmation dialog: "Unsaved changes. Save before closing?" with Save / Discard / Cancel buttons. |

---

## App 2: Live Monitor

### Purpose

Real-time mixing-time detection during experiments. Loads `vessel_rois.json`, captures live video, crops per-vessel feeds, runs analysis in parallel, detects mixing completion.

### Launch

```bash
python -m mixing_monitor.monitor.monitor_app [--config path/to/vessel_rois.json]
```

### Window Specifications

- **Title:** "Live Mixing Monitor — Kineticolor"
- **Default size:** 1400 × 800 px (expands to fit 4 vessels)
- **Resizable:** Yes
- **Minimum size:** 1000 × 600 px

### Layout

The window has three horizontal sections stacked vertically: vessel cards (top, takes most space), results bar (middle), and keyboard legend (bottom).

#### Vessel Cards Section

Always renders a **4-slot grid** (2×2 or 1×4 depending on window width). Active vessels show live crops. Inactive slots show placeholder cards.

**With 3 active vessels:**

```
┌──────────────────────────────────────────────────────────────────────┐
│  Live Mixing Monitor — Kineticolor                          [─][□][×]│
├──────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌─── V1 (Green) ───────┐  ┌─── V2 (Blue) ────────┐                │
│  │                       │  │                       │                │
│  │    Live cropped       │  │    Live cropped       │                │
│  │    video feed         │  │    video feed         │                │
│  │                       │  │                       │                │
│  │                       │  │                       │                │
│  ├───────────────────────┤  ├───────────────────────┤                │
│  │  a*: 24.7   ΔE: 12.3 │  │  a*: 31.2   ΔE: 8.4  │                │
│  │  ▁▂▃▅▇▇▇▇▇▇▇▇▇▇▇██  │  │  ▁▃▅▇▇▇▇▇▇▇▇███████  │                │
│  └───────────────────────┘  └───────────────────────┘                │
│                                                                      │
│  ┌─── V3 (Yellow) ──────┐  ┌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┐                │
│  │                       │  ╎                        ╎                │
│  │    Live cropped       │  ╎      No Vessel         ╎                │
│  │    video feed         │  ╎                        ╎                │
│  │                       │  ╎                        ╎                │
│  │                       │  ╎                        ╎                │
│  ├───────────────────────┤  ╎                        ╎                │
│  │  a*: 18.9   ΔE: 15.1 │  ╎                        ╎                │
│  │  ▁▂▄▅▆▇▇▇▇▇▇▇▇▇▇▇█  │  ╎                        ╎                │
│  └───────────────────────┘  └╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌┘                │
│                                                                      │
├──────────────────────────────────────────────────────────────────────┤
│  V1: ● Idle              V2: ◉ Mixing (4.2s)                        │
│  V3: ✓ Complete 12.4s    V4: ○ No vessel                             │
├──────────────────────────────────────────────────────────────────────┤
│  [1-4] Arm vessel   [R] Reset all   [S] Save session   [Q] Quit     │
└──────────────────────────────────────────────────────────────────────┘
```

### Vessel Card Widget (`vessel_card.py`)

Each card is a self-contained widget with these regions:

#### Active Card (vessel present in config)

| Region | Content |
|--------|---------|
| **Header bar** | Vessel label + color indicator. Height: 28px. Background: vessel color at 15% opacity. Text: vessel label in vessel color, bold. |
| **Video area** | Live cropped feed, aspect-ratio preserved, black letterboxing. Takes majority of card height. |
| **Metrics bar** | Two key numbers displayed: current mean a* value and current grand ΔE. Font: monospace, 13px. Updated every analysis frame. |
| **Sparkline** | `pyqtgraph.PlotWidget` showing the last 150 data points of the primary tracked metric (mean a*). Height: 40px fixed. No axes, no labels — just the line. Line color matches vessel color. Background transparent. |
| **State overlay** | Semi-transparent overlay on the video area reflecting current state (see Vessel States below). |

#### Inactive Card (vessel not in config)

| Region | Content |
|--------|---------|
| **Entire card** | 2px dashed border in `#9ca3af` (gray-400). Background: `#f9fafb` (gray-50). Centered text: "No Vessel" in `#9ca3af`, 14px. No metrics, no sparkline. |

### Vessel States

Each vessel independently follows this state machine:

```
                             ┌──────────────────────────────┐
                             │                              │
                             ▼                              │
  ○ Disabled ──(not in config, card shows "No Vessel")      │
                                                            │
  ● Idle ──[press N]──▶ ◉ Armed ──(a* spike)──▶ ◉ Mixing ──┤
     ▲                                              │       │
     │                                   (plateau)  │       │
     │                                              ▼       │
     └──────────[press N again]──────── ✓ Complete ─┘
```

| State | Visual Treatment | Behavior |
|-------|-----------------|----------|
| **Disabled** | Dashed border, "No Vessel" text | No processing. |
| **Idle** | Normal card, no overlay | Video and metrics display live but no plateau detection. Waiting for user to arm. |
| **Armed** | Pulsing colored border (vessel color, 2px→4px, 1Hz) | The current frame is captured as the **reference frame** for this vessel. Plateau detection begins. System watches for mean a* to spike above threshold (pink introduced). |
| **Mixing** | Colored border stays solid 3px. Elapsed timer shown in metrics bar. | Mean a* is being tracked. Sparkline updates live. The system is watching for a* to drop and plateau (pink → transparent). |
| **Complete** | Green checkmark overlay on video (semi-transparent). Mixing time shown prominently. | Final mixing time displayed. Sparkline frozen. Press the vessel's key again to re-arm (goes back to Idle → Armed on next press). |

### State Transitions — Detail

**Idle → Armed (user presses vessel key 1–4):**
- Capture the current cropped frame as the reference for this vessel.
- Reset the sparkline buffer.
- Reset elapsed timer to 0.
- Start recording analysis data for this vessel.

**Armed → Mixing (automatic):**
- Triggered when mean a* of the crop exceeds `armed_trigger_threshold` (default: 5.0 above the reference frame's mean a*). This detects the moment pink indicator is added to the vessel.
- Start the elapsed timer.

**Mixing → Complete (automatic):**
- Triggered when the **rolling standard deviation** of mean a* over the last `plateau_window` readings (default: 60 readings ≈ 2 seconds at 30fps) falls below `plateau_std_threshold` (default: 0.5).
- AND the current mean a* is within `plateau_return_threshold` (default: 3.0) of the reference frame's mean a* (confirming the liquid has gone transparent again).
- Record `mixing_time = elapsed seconds since Armed → Mixing transition`.

**Complete → Idle (user presses vessel key again):**
- Clear the overlay and results.
- Go back to Idle, ready for next trial.

### Analysis Pipeline — Per Frame

```
Capture Thread (30fps) ─── grabs latest frame ───▶ shared_frame buffer (with Lock)
                                                          │
Main Thread timer (fires every 33ms ≈ 30Hz) ──────────────┘
    │
    ├── read shared_frame
    ├── crop ROI for vessel 1 ──▶ submit to ThreadPoolExecutor
    ├── crop ROI for vessel 2 ──▶ submit to ThreadPoolExecutor
    ├── crop ROI for vessel 3 ──▶ submit to ThreadPoolExecutor
    ├── crop ROI for vessel 4 ──▶ submit to ThreadPoolExecutor
    │
    ├── collect futures (with timeout)
    │
    ├── for each vessel:
    │     ├── update sparkline data
    │     ├── update metrics display
    │     ├── check state transition
    │     └── update state overlay
    │
    └── update video displays (QImage from crop)
```

### Capture Thread (`capture_thread.py`)

A `QThread` that runs a tight loop:

```python
class CaptureThread(QThread):
    """Dedicated thread for frame grabbing. Drops stale frames."""

    frame_ready = pyqtSignal()  # emitted when a new frame is available

    def __init__(self, device_index: int, resolution: tuple[int, int]):
        ...
        self._frame = None
        self._lock = threading.Lock()
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self._device_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._resolution[1])
        self._running = True

        while self._running:
            ret, frame = cap.read()
            if ret:
                with self._lock:
                    self._frame = frame  # always overwrite (drop stale)
                self.frame_ready.emit()

        cap.release()

    def get_latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._frame.copy() if self._frame is not None else None

    def stop(self):
        self._running = False
        self.wait()
```

### Per-Vessel Analyzer (`vessel_analyzer.py`)

Stateless per-frame computation. The monitor's main loop manages state.

```python
class VesselAnalyzer:
    """Computes mixing metrics for a single vessel crop."""

    def __init__(self, reference_frame: np.ndarray):
        """
        Args:
            reference_frame: BGR crop of the vessel at the reference time.
                             Converted to Lab internally and stored.
        """
        self._ref_lab = cv2.cvtColor(reference_frame, cv2.COLOR_BGR2Lab).astype(np.float32)
        self._ref_mean_a = float(self._ref_lab[:, :, 1].mean())

    def analyze(self, frame_bgr: np.ndarray) -> dict:
        """Analyze a single frame crop.

        Returns:
            {
                "mean_a_star": float,    # mean of a* channel
                "mean_delta_e": float,   # grand delta-E from reference
                "timestamp": float,      # not set here, set by caller
            }
        """
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab).astype(np.float32)
        mean_a = float(lab[:, :, 1].mean())
        delta_e = float(np.sqrt(((lab - self._ref_lab) ** 2).sum(axis=2)).mean())
        return {"mean_a_star": mean_a, "mean_delta_e": delta_e}

    @property
    def reference_mean_a(self) -> float:
        return self._ref_mean_a
```

### ThreadPoolExecutor Coordination (`analysis_pool.py`)

```python
class AnalysisPool:
    """Manages parallel analysis of up to 4 vessel crops."""

    def __init__(self, max_workers: int = 4):
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._analyzers: dict[int, VesselAnalyzer] = {}  # vessel_id -> analyzer

    def set_reference(self, vessel_id: int, reference_crop: np.ndarray) -> None:
        """Set/reset the reference frame for a vessel."""
        self._analyzers[vessel_id] = VesselAnalyzer(reference_crop)

    def submit_all(self, crops: dict[int, np.ndarray]) -> dict[int, Future]:
        """Submit crops for all active vessels. Returns {vessel_id: Future}."""
        futures = {}
        for vid, crop in crops.items():
            if vid in self._analyzers:
                futures[vid] = self._pool.submit(self._analyzers[vid].analyze, crop)
        return futures

    def shutdown(self) -> None:
        self._pool.shutdown(wait=False)
```

### Results Bar (`results_bar.py`)

A horizontal bar at the bottom showing one line per vessel slot (always 4 lines):

| Vessel State | Display |
|--------------|---------|
| Disabled | `V4: ○ No vessel` (gray text) |
| Idle | `V1: ● Idle — Press [1] to arm` (white/default text) |
| Armed | `V1: ◉ Armed — Waiting for indicator...` (vessel color text, pulsing dot) |
| Mixing | `V2: ◉ Mixing — 4.2s elapsed, a*: 18.3` (vessel color text) |
| Complete | `V3: ✓ Complete — Mixed in 12.4s` (green text, bold time) |

### Keyboard Controls

All keyboard input is handled by the main window's `keyPressEvent`. Active only when the window has focus.

| Key | Action | Conditions |
|-----|--------|------------|
| `1` | Arm/reset vessel 1 | Only if vessel 1 is in config. Idle→Armed, or Complete→Idle. |
| `2` | Arm/reset vessel 2 | Only if vessel 2 is in config. |
| `3` | Arm/reset vessel 3 | Only if vessel 3 is in config. |
| `4` | Arm/reset vessel 4 | Only if vessel 4 is in config. |
| `R` | Reset all vessels to Idle | Clears all state, sparklines, timers. Asks confirmation if any vessel is in Mixing state. |
| `S` | Save session log | Writes current session data to CSV. See Session Logging. |
| `Q` | Quit | Asks confirmation if any vessel is in Mixing state. |

### Session Logging (`session_logger.py`)

When the user presses `S`, or when a vessel transitions to Complete, data is appended to a session CSV file.

**File location:** `mixing_monitor/results/session_YYYYMMDD_HHMMSS.csv` (created on monitor app launch).

**CSV columns:**

```
timestamp, vessel_id, vessel_label, event, mixing_time_s, mean_a_star_ref, mean_a_star_final, mean_delta_e_final, notes
```

**Events logged:**

| Event | When |
|-------|------|
| `armed` | Vessel transitions to Armed state |
| `mixing_start` | Vessel transitions to Mixing state (indicator detected) |
| `mixing_complete` | Vessel transitions to Complete state (plateau detected) |
| `session_save` | User presses S (one row per active vessel with current state snapshot) |

Additionally, per-vessel time-series data (every analyzed frame's mean_a* and mean_delta_e) is saved to separate files when a vessel completes:

```
mixing_monitor/results/session_YYYYMMDD_HHMMSS_V1_timeseries.csv
mixing_monitor/results/session_YYYYMMDD_HHMMSS_V2_timeseries.csv
...
```

These can be loaded into the main Kineticolor GUI for post-hoc 6-metric analysis if the video was also recorded.

---

## Analysis Parameters

### Defaults (hardcoded in `constants.py`, overridable via CLI flags later)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `armed_trigger_threshold` | 5.0 | a* must rise this far above reference mean a* to trigger Mixing. |
| `plateau_window` | 60 | Number of consecutive readings for plateau check (≈2s at 30fps). |
| `plateau_std_threshold` | 0.5 | Rolling std of mean a* must fall below this for plateau. |
| `plateau_return_threshold` | 3.0 | Current mean a* must be within this of reference a* to confirm transparent. |
| `sparkline_buffer_size` | 150 | Number of data points shown in sparkline. |
| `analysis_fps_target` | 30 | Target analysis rate. If analysis is slower, frames are skipped gracefully. |

### Why These Defaults

- **armed_trigger_threshold = 5.0**: A phenolphthalein pink produces a* shifts of 15–30 units in Lab space. 5.0 avoids false triggers from lighting flicker while still catching the indicator addition quickly.
- **plateau_window = 60**: At 30fps, this is 2 seconds. Mixing times for these benchtop reactors are typically 5–30 seconds, so a 2-second confirmation window is short enough to not add excessive lag to the detection.
- **plateau_std_threshold = 0.5**: Lab a* noise on a stable transparent liquid under fluorescent lighting is typically 0.1–0.3 units. 0.5 gives headroom without being so loose that partial mixing is declared complete.
- **plateau_return_threshold = 3.0**: Confirms the liquid has returned to near-transparent (near reference a*). Prevents declaring "complete" if mixing plateaus at a still-pink intermediate state.

---

## Edge Cases — Monitor App

| Scenario | Behavior |
|----------|----------|
| **No config file found** | On launch, show dialog: "No vessel config found. Run the Calibration app first." with an "Open Calibration" button that launches the calibration app and closes the monitor. |
| **Config file camera device not available** | Show warning: "Camera 'Camo (iPhone)' (index 1) not found. Available devices: [list]. Select one or reconnect." Dropdown to pick alternative device. ROIs still load (they're pixel coordinates, resolution-dependent — if resolution differs, scale them and show warning). |
| **Camera disconnects during monitoring** | Video feeds freeze. Status bar shows "⚠ Camera disconnected." Any vessel in Mixing state keeps its timer running (mixing doesn't stop because the camera died). When camera reconnects (checked every 2 seconds by capture thread), feeds resume automatically. |
| **Camera frame resolution differs from config** | Scale all ROIs proportionally. Show a one-time warning banner: "Camera resolution changed from 1280×720 to 1920×1080. ROIs scaled proportionally. Verify positions in Calibration app if needed." |
| **Vessel key pressed for non-existent vessel** | Ignored. No visible feedback. |
| **Vessel key pressed during Mixing** | Ignored. Must wait for Complete, then press again to re-arm. This prevents accidental reset of a running experiment. |
| **All vessels complete** | Results bar shows all complete times. Nothing auto-resets. User explicitly presses keys to re-arm for next trial. |
| **Analysis is slower than 30fps** | Frame dispatcher uses the latest available frame (frame-dropping). The sparkline and metrics update at whatever rate analysis achieves. Display shows effective analysis FPS in the title bar: "Live Mixing Monitor — 28fps". |
| **Very short mixing time (<1s)** | Still detected and reported. The plateau window shrinks if fewer than `plateau_window` readings exist (minimum 15 readings ≈ 0.5s). |
| **Very long mixing time (>60s)** | No timeout. The system keeps tracking indefinitely until plateau is detected or user resets. |
| **Lighting change mid-experiment** | Will affect mean a* baseline. The system compares against the reference frame captured at arm time, so gradual lighting shifts may cause false plateaus or missed detections. Mitigation: arm the vessel as close to the moment of indicator addition as possible. Document this as a known limitation. |
| **Multiple vessels armed simultaneously** | Fully supported. Each vessel has independent state, reference frame, timer, and plateau detection. The ThreadPoolExecutor handles all in parallel. |
| **User presses S (save) before any experiment** | Session file is created with header row only. No error. |
| **App is closed while vessel is Mixing** | Confirmation dialog: "Vessel V2 is currently being monitored. Quit anyway?" Yes/No. If Yes, session log is flushed to disk before exit. |
| **ROI extends to area that is sometimes occluded** | (e.g., researcher's hand passes in front of vessel) Mean a* will spike briefly. The plateau detection's rolling window smooths this out — a momentary occlusion of <2s will not trigger false completion. Document as known limitation: sustained occlusion (>2s) during mixing may cause false results. |

---

## UI Style Guidelines

- **Font:** System default (Segoe UI on Windows). Monospace for metric values (Consolas on Windows).
- **Colors:** Dark theme is NOT required. Use the default PyQt6 system palette (light theme). Vessel colors as specified in the color mapping table.
- **Spacing:** 8px base unit. Margins: 12px. Card gaps: 12px.
- **Icons:** Use Unicode characters for state indicators (●, ◉, ✓, ○, ⚠). No icon library dependencies.
- **Borders:** Active cards: 2px solid in vessel color. Inactive cards: 2px dashed in gray-400 (`#9ca3af`).
- **No custom title bars.** Use native Windows window chrome.
- **No splash screen.** Both apps should open to functional state in under 2 seconds.

---

## Performance Targets

| Metric | Target | Hard Limit |
|--------|--------|------------|
| Frame capture → display latency | <50ms | <100ms |
| Per-vessel analysis time | <10ms | <20ms |
| End-to-end (frame capture → metrics update) | <70ms | <150ms |
| Memory usage (3 vessels, 720p) | <300MB | <500MB |
| CPU usage (3 vessels, 720p, i7) | <40% | <70% |

If analysis cannot keep up with 30fps, the system drops frames gracefully (always analyze the most recent frame, skip stale ones). The UI must never freeze or stutter.

---

## Testing Plan

### Unit Tests

| Module | Tests |
|--------|-------|
| `camera.py` | Device enumeration returns list format. `open()` fails gracefully on invalid index. Resolution fallback works. |
| `roi_config.py` | Save/load round-trips correctly. Schema validation rejects bad data. ROI scaling on resolution change. |
| `vessel_analyzer.py` | Returns correct dict keys. Delta-E is 0 when frame equals reference. a* increases with pink frames. |
| `analysis_pool.py` | Submits and collects futures for 1–4 vessels. Handles missing analyzer gracefully. |
| `session_logger.py` | Creates file with correct headers. Appends events. Handles concurrent writes. |

### Integration Tests

| Test | Description |
|------|-------------|
| Calibration save → Monitor load | Write config with calibration app, verify monitor app reads it correctly and creates correct crops. |
| Full pipeline with synthetic video | Feed a known sequence of frames (pink → transparent) through the analyzer. Verify mixing time detection within 0.5s of ground truth. |
| State machine transitions | Verify all transitions: Idle→Armed→Mixing→Complete→Idle. Verify blocked transitions (key press during Mixing). |

### Manual Testing Checklist

- [ ] Camera connect/disconnect/reconnect on both apps
- [ ] Draw, move, resize, delete ROIs in calibration
- [ ] Save and load config files
- [ ] Arm, detect, complete for each vessel independently
- [ ] Arm all vessels simultaneously
- [ ] Session log contains all expected events
- [ ] App closes cleanly during Mixing (with confirmation)
- [ ] Monitor launches without config (shows error + calibration link)

---

## Dependencies (complete `mixing_monitor/requirements.txt`)

```
opencv-python>=4.8.0
numpy>=1.24.0
PyQt6>=6.5.0
pyqtgraph>=0.13.3
pygrabber>=0.2          # Windows DirectShow device enumeration
```

This is a **self-contained** requirements file. It does not reference or depend on the main Kineticolor `requirements.txt`. Install with:

```bash
pip install -r mixing_monitor/requirements.txt
```

All packages must be installed BEFORE going to the lab. Verify offline operation by running both apps with Wi-Fi disabled.

---

## Out of Scope (v1)

These are explicitly NOT included in this version:

- **Full 6-metric Kineticolor analysis in real-time.** Only mean a* and grand ΔE are computed live. Full analysis is done post-hoc on saved video.
- **Video recording.** The monitor app does not record the camera feed. Use the phone's native recording or a separate tool (OBS) if video archival is needed.
- **Automatic camera angle detection or vessel detection.** ROIs are manually drawn and fixed.
- **Network streaming or remote monitoring.** Everything is local to the one Windows laptop.
- **Dark mode UI.**
- **Multi-camera support.** One camera only.
- **Post-hoc video replay in the monitor app.** Use the existing Kineticolor GUI for that.
- **Automatic report generation.**
- **Phone-as-camera without virtual webcam driver.** The virtual webcam software (Camo or EpocCam) must be installed separately by the user.

---

## Appendix: Wired iPhone Setup Instructions

Include this in the app's Help menu or as a bundled `SETUP_CAMERA.md`:

1. **On iPhone:** Install "Camo" from the App Store (free tier is sufficient for 720p).
2. **On Windows laptop:** Download and install "Camo Studio" from https://reincubate.com/camo/ (do this while you have internet access).
3. **Connect iPhone to laptop via USB cable** (Lightning or USB-C depending on model).
4. **Open Camo Studio on Windows.** It should detect the iPhone within a few seconds.
5. **In Camo Studio:** Set resolution to 720p, frame rate to 30fps.
6. **Verify:** Open the Calibration app. The dropdown should show "Camo" as an available device.
7. **Note:** Camo does NOT require internet after initial install. It works fully offline over USB.

Alternative: **EpocCam** follows a similar flow (phone app + desktop driver). **DroidCam** is Android-only and not applicable for iPhone.
