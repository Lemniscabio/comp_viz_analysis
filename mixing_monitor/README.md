# Mixing Monitor — Real-Time Multi-Vessel Mixing Detection

A standalone PyQt6 Windows desktop application for real-time mixing-time detection in bioreactor vessels. Part of the Kineticolor project. Works fully offline once installed.

---

## Contents

1. [Fresh Windows Setup](#1-fresh-windows-setup)
2. [Camera Setup — iPhone](#2a-camera-setup--iphone-recommended)
3. [Camera Setup — Android](#2b-camera-setup--android)
4. [Running the App](#3-running-the-app)
5. [Calibration Workflow](#4-calibration-workflow)
6. [Live Monitor Workflow](#5-live-monitor-workflow)
7. [Keyboard Shortcuts](#keyboard-shortcuts)
8. [Exporting Results](#exporting-results)
9. [Troubleshooting](#troubleshooting)

---

## 1. Fresh Windows Setup

> Do steps 1.1–1.4 **before going to the lab** — an internet connection is required.

### 1.1 Install Python

1. Go to [python.org/downloads](https://www.python.org/downloads/) and download **Python 3.12** (or newer).
2. Run the installer. On the first screen, **tick "Add Python to PATH"** before clicking Install.
3. Verify: open **Command Prompt** (`Win + R` → type `cmd` → Enter) and run:
   ```
   python --version
   ```
   Expected output: `Python 3.12.x`

> Use the **python.org** installer, not the Microsoft Store version. The Store version can have PATH issues that break pip.

### 1.2 Download the project

If you have Git installed:
```
git clone <repo-url>
cd comp_viz_analysis
```

Otherwise, download the ZIP from your repository host and extract it. The folder you want is `comp_viz_analysis\`.

### 1.3 Create a virtual environment (recommended)

Open Command Prompt in the `comp_viz_analysis` folder:
```
python -m venv .venv
.venv\Scripts\activate
```

You should see `(.venv)` at the start of your prompt. Run all subsequent commands in this same window.

### 1.4 Install dependencies

```
pip install -r mixing_monitor/requirements.txt
```

This installs: OpenCV, NumPy, PyQt6, pyqtgraph, and pygrabber. Takes 1–3 minutes.

Verify the install worked:
```
python -c "import cv2, PyQt6, pyqtgraph; print('OK')"
```

---

## 2a. Camera Setup — iPhone (Recommended)

The app needs your phone to appear as a **virtual webcam** on Windows. Camo does this over USB (no WiFi needed, works in a Faraday cage or offline lab).

### Install (requires internet — do before the lab)

1. **On iPhone:** Install **Camo** from the App Store (free tier gives 720p).
2. **On Windows:** Download and install **Camo Studio** from [reincubate.com/camo](https://reincubate.com/camo/).

### Use

1. Connect iPhone to laptop with a Lightning or USB-C cable.
2. Open **Camo Studio** on Windows. It will show a live preview.
3. In Camo Studio, set **Resolution: 720p**, **Frame Rate: 30 fps**.
4. Leave Camo Studio open in the background while running the mixing monitor.
5. In the app's camera dropdown, select **Camo** (or the device index it appears at).

**EpocCam** (Elgato) is an alternative that follows the same USB flow and also appears as a DirectShow virtual webcam.

---

## 2b. Camera Setup — Android

Two options depending on your connection preference.

### Option A — DroidCam (USB or WiFi)

DroidCam installs a virtual webcam driver on Windows, making the phone's camera appear exactly like a USB webcam.

1. **On Android:** Install **DroidCam** from Google Play.
2. **On Windows:** Download and install the **DroidCam Windows Client** from [dev47apps.com](https://www.dev47apps.com/). This installs the required virtual webcam driver.
3. Connect your phone:
   - **USB:** Enable USB debugging on Android (`Settings → Developer Options → USB Debugging`), connect cable, open DroidCam app and Windows Client — click **Start**.
   - **WiFi:** Both devices on the same network. Open DroidCam app, note the IP address shown, enter it in the Windows Client.
4. The phone camera now appears as a DirectShow device (webcam) in the app's dropdown.

Free tier: 640×480. The paid upgrade ($5 one-time) unlocks 1080p HD — recommended for accurate color detection.

### Option B — EpocCam (USB or WiFi)

Same flow as the iPhone Camo setup:

1. **On Android:** Install **EpocCam** from Google Play.
2. **On Windows:** Install **EpocCam Drivers** from Elgato's website.
3. Connect via USB or WiFi and select EpocCam in the app's camera dropdown.

> **Do not use IP Webcam or similar WiFi-only streaming apps.** Those stream MJPEG/RTSP over a URL and do not create a virtual webcam device. The mixing monitor requires a DirectShow device (integer index), not a network stream URL.

---

## 3. Running the App

From the `comp_viz_analysis` folder (with your virtual environment active if you created one):

```
python -m mixing_monitor.main_app
```

This launches a single window. If no saved calibration exists, it opens the **Calibration** screen first. If a calibration was saved previously, it goes straight to the **Live Monitor**.

Optional — specify a different config file:
```
python -m mixing_monitor.main_app --config path\to\vessel_rois.json
```

---

## 4. Calibration Workflow

Run once per physical camera setup (re-run if you move the camera or vessels).

1. Select your camera from the dropdown (auto-selected if only one device is found).
2. Choose the number of vessels (1–4) using the radio buttons.
3. **Draw bounding boxes** by clicking and dragging on the live video feed:
   - Drag **inside** a box to move it.
   - Drag a **corner or edge handle** to resize it.
   - **Right-click** a box to delete it.
4. Each box should tightly frame the liquid region of one vessel, excluding clamps, labels, and the vessel wall.
5. Click **Save Config** when all boxes are placed correctly.

Config is saved to `mixing_monitor/config/vessel_rois.json`. The app switches to the Live Monitor automatically after saving.

---

## 5. Live Monitor Workflow

### Per-experiment steps

1. Position your camera to frame all vessels as they were during calibration.
2. Launch the app. All configured vessels show live video crops.
3. **Just before adding the pH indicator** to a vessel, press its number key (1–4) to **Arm** it.
   - The current frame is saved as the reference (baseline).
   - The card border pulses to indicate armed status.
4. **Add the indicator** (e.g., phenolphthalein drops, HCl). The system automatically detects the color change and transitions to **Mixing** state, starting the elapsed timer.
5. When the liquid returns to its final uniform color (mixing complete), the system detects stabilization and displays the **Estimated Mixing time**.
6. Press the vessel number key again to re-arm for the next trial, or press `R` to reset all vessels.

### Tips

- **Arm as late as possible** — arm the vessel in the 1–2 seconds just before adding the indicator, not 30 seconds before. The reference frame is taken at arm time; lighting changes after arming bias the ΔE signal.
- **Keep lighting constant** throughout the experiment. The system does not correct for illumination changes.
- **Multiple vessels simultaneously:** arm each vessel independently with its number key. Analysis runs in parallel.

---

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1` | Arm vessel 1 (or prompt to force-stop if already armed/mixing) |
| `2` | Arm vessel 2 (same) |
| `3` | Arm vessel 3 (same) |
| `4` | Arm vessel 4 (same) |
| `R` | Reset all vessels to Idle (prompts for confirmation if any are mixing) |
| `E` | Export results to CSV (opens file save dialog) |
| `Q` | Quit |

---

## Exporting Results

Press **`E`** or click the **Export CSV** button in the bottom bar at any time. A file save dialog opens — choose your destination and filename.

The CSV contains two sections:

**Time series** (one row per analysis frame, per vessel):

| Column | Description |
|--------|-------------|
| `vessel_id` | 1–4 |
| `label` | V1–V4 |
| `elapsed_s` | Seconds since arm key was pressed |
| `mean_a_star` | Mean CIE a* across the vessel ROI |
| `delta_e` | Grand Delta-E from reference frame |
| `pink_fraction` | Fraction of pixels above the pink threshold |

**Summary block** (appended after the time series):

| Column | Description |
|--------|-------------|
| `pipetting_delay_s` | Seconds between arm key press and ΔE trigger (HCl addition delay) |
| `mixing_time_s` | Estimated mixing time (from ΔE trigger to stable completion) |

---

## Troubleshooting

### Camera shows black frames or doesn't appear in the dropdown

- Make sure the camera app (Camo Studio / DroidCam) is open and the phone is connected **before** launching the mixing monitor.
- In Camo Studio, ensure the preview is live (not a still image or black screen). Toggle the connection off/on in the Camo app.
- Only one application can hold the DirectShow camera device at a time. Close any other app using the camera (Teams, Zoom, browser).
- If the issue persists after Calibrate Again, wait a few seconds before the calibration screen's camera starts — the app waits 1.2 s before re-opening the device to avoid a Windows DirectShow double-open bug.

### "No camera found" in calibration dropdown

- Check Device Manager (`Win + X → Device Manager → Cameras`) — the virtual webcam must appear there.
- Reinstall the Camo Studio / DroidCam Windows driver.
- Try a different USB port or cable.

### ΔE trigger fires immediately on arm (false start)

- The vessel ROI is capturing a region with changing lighting or moving objects (e.g., a fan reflection, stirrer vortex). Tighten the ROI to the liquid bulk and exclude the surface.
- Arm the vessel closer to the moment of indicator addition.

### Completion detected too early (false plateau)

- Increase `DELTA_E_STABILITY_THRESHOLD` in `common/constants.py` to require a stricter (lower) std before declaring complete. See `Methodology.md` for the full tuning table.

### Completion never detected (mixing runs forever)

- Decrease `DELTA_E_STABILITY_THRESHOLD` or `DELTA_E_STABILITY_WINDOW`. The color may be genuinely stable but the threshold is set too tight.
- Check that pink detection is not blocking completion: if `PINK_FRACTION_COMPLETE` is too low for your indicator residual color, raise it slightly.

### App window is too small / vessel cards overlap the results bar

- Drag the window larger, or maximize it. Minimum recommended window height is 760 px.
