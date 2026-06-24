# Claude Code Prompt — Real-Time Mixing Monitor

Paste everything below the line into the Claude Code terminal.

---

Read the PRD at `mixing_monitor/PRD_RealTime_Mixing_Monitor.md` carefully before writing any code. That document is the single source of truth for architecture, UI specs, edge cases, and file structure. Every implementation decision should trace back to it.

## What to build

A PyQt6 Windows desktop application for real-time mixing-time detection in bioreactor vessels. It consists of two separate apps (calibration + live monitor) that communicate through a shared JSON config file. All code goes inside the `mixing_monitor/` folder at the project root. Do NOT create, modify, or import from anything outside `mixing_monitor/`.

## Build order

Follow this sequence strictly. Finish and test each phase before starting the next.

### Phase 1 — `common/` layer (no GUI, no PyQt6)

1. `common/constants.py` — all shared constants, vessel color map, default thresholds, file paths.
2. `common/camera.py` — `CameraSource` class with device enumeration (pygrabber with OpenCV fallback), open/read/release, resolution negotiation. Must work headless.
3. `common/roi_config.py` — load/save/validate `vessel_rois.json`. Include ROI scaling logic for resolution mismatches. Schema validation that rejects malformed configs.
4. `common/vessel_analyzer.py` — `VesselAnalyzer` class. Takes a BGR reference frame, exposes `analyze(frame_bgr) -> dict` returning `mean_a_star` and `mean_delta_e`. Pure OpenCV + NumPy, no imports from `src/`.

Verify: each module should be importable and testable independently with a simple script.

### Phase 2 — Calibration app

5. `calibration/roi_canvas.py` — the hardest widget. Interactive rectangle drawing on a live video feed. Must support: click-drag to draw, drag inside to move, drag corners/edges to resize (with visible handles), right-click to delete, 1-4 color-coded rectangles, minimum 50x50px, clamp to frame bounds. Coordinates stored in camera pixel space, not widget space.
6. `calibration/calibration_window.py` — full window layout per the PRD wireframe: camera dropdown + refresh, vessel count radio buttons (1-4), ROI canvas, save/load buttons, status messages.
7. `calibration/calibration_app.py` — entry point with `--config` flag. Launches QApplication + CalibrationWindow.

Verify: launch the calibration app, select a camera (or test with a video file if no camera available), draw rectangles, save config, reload config and confirm ROIs restore correctly.

### Phase 3 — Monitor app

8. `monitor/capture_thread.py` — QThread that grabs frames in a tight loop, overwrites a shared buffer (frame-dropping design), emits `frame_ready` signal.
9. `monitor/analysis_pool.py` — `AnalysisPool` wrapping `ThreadPoolExecutor(max_workers=4)`. Methods: `set_reference()`, `submit_all()`, `shutdown()`.
10. `monitor/vessel_card.py` — single vessel card widget per the PRD spec. Active state: header bar + live video + metrics bar + pyqtgraph sparkline. Inactive state: dashed border + "No Vessel" text.
11. `monitor/results_bar.py` — horizontal status bar showing all 4 vessel states with Unicode indicators.
12. `monitor/session_logger.py` — CSV writer for events and per-vessel timeseries. Creates `mixing_monitor/results/` directory on first write.
13. `monitor/monitor_window.py` — full window layout: 2x2 vessel card grid (always 4 slots), results bar, keyboard legend. Implements `keyPressEvent` for 1-4/R/S/Q. Wires up capture thread → crop → analysis pool → UI update loop. Implements the vessel state machine (Idle → Armed → Mixing → Complete).
14. `monitor/monitor_app.py` — entry point with `--config` flag. Shows error + calibration launch button if no config found.

### Phase 4 — Polish

15. `mixing_monitor/requirements.txt` — exact list from the PRD.
16. `mixing_monitor/README.md` — setup instructions including Camo/iPhone wired setup, how to run calibration, how to run monitor, keyboard shortcuts reference.
17. Handle every edge case listed in the PRD's edge case tables (camera disconnect/reconnect, resolution mismatch scaling, missing config, unsaved changes confirmation, quit-during-mixing confirmation, etc.).

## Hard constraints

- **Offline only.** Zero network calls at runtime. No CDN fonts, no API calls, no telemetry. Everything works with Wi-Fi disabled.
- **Self-contained.** All files inside `mixing_monitor/`. Zero imports from `src/`, `config/`, `scripts/`, or anywhere else in the repo. The ~10 lines of Lab conversion + Delta-E math are reimplemented locally in `vessel_analyzer.py`.
- **No modifications outside `mixing_monitor/`.** Do not touch `src/`, `config/default_config.yaml`, the root `requirements.txt`, or any other existing file.
- **Windows target.** The app runs on Windows 10/11. Use `pygrabber` for DirectShow device enumeration with an OpenCV index-probing fallback if pygrabber fails to import.
- **PyQt6 only for GUI.** No Tkinter, no web UI, no Electron. System theme (light mode, Segoe UI font). No custom title bars.
- **Threading, not multiprocessing** for parallel vessel analysis. OpenCV/NumPy release the GIL so threads achieve true parallelism. See the PRD's architecture section for rationale.
- **Frame-dropping capture design.** The capture thread always overwrites the latest frame. The analysis loop reads whatever is newest. Never queue frames — latency matters more than completeness.

## What NOT to build

Do not implement anything listed in the PRD's "Out of Scope (v1)" section: no full 6-metric analysis, no video recording, no dark mode, no multi-camera, no automatic vessel detection, no network streaming, no post-hoc replay.

## When in doubt

Refer back to the PRD. It has class-level API contracts, state machine diagrams, UI wireframes, edge case tables, and performance targets. If something isn't specified in the PRD, make the simplest choice that works and leave a `# TODO:` comment noting the assumption.
