"""Real-time metric plots panel using pyqtgraph."""
from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import pyqtgraph as pg
from PyQt6.QtWidgets import QCheckBox, QComboBox, QGridLayout, QHBoxLayout, QVBoxLayout, QWidget


class PlotsPanel(QWidget):
    """Six real-time metric plots in a 3x2 grid."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        pg.setConfigOptions(antialias=True)

        self._timestamps: List[float] = []
        self._data: Dict[str, List[float]] = {
            "grand_delta_e": [],
            "contrast": [],
            "homogeneity": [],
            "energy": [],
            "contact_perimeter": [],
            "variance_r": [],
            "variance_g": [],
            "variance_b": [],
            "variance_l": [],
            "variance_a": [],
            "variance_b_star": [],
            "variance_delta_e": [],
        }
        self._row_avg_data: List[np.ndarray] = []
        self._col_avg_data: List[np.ndarray] = []
        self._grid_rows: int = 5
        self._grid_cols: int = 5

        main_layout = QVBoxLayout(self)

        de_header = QHBoxLayout()
        self._de_mode = QComboBox()
        self._de_mode.addItems(["Grand Delta-E", "By Row", "By Column"])
        self._de_mode.currentIndexChanged.connect(self._on_de_mode_changed)
        de_header.addWidget(self._de_mode)

        self._de_normalize = QCheckBox("Normalize (0-1)")
        self._de_normalize.setEnabled(False)
        self._de_normalize.setToolTip(
            "Show Delta-E normalized to 0-1 range\n"
            "(divided by maximum value). Available after analysis completes."
        )
        self._de_normalize.toggled.connect(self._update_de_plot)
        de_header.addWidget(self._de_normalize)

        de_header.addStretch()
        main_layout.addLayout(de_header)

        grid = QGridLayout()
        main_layout.addLayout(grid)

        self._plot_de = pg.PlotWidget(title="Delta-E (Color Distance)")
        self._plot_contrast = pg.PlotWidget(title="Contrast")
        self._plot_energy = pg.PlotWidget(title="Energy / ASM")
        self._plot_homogeneity = pg.PlotWidget(title="Homogeneity")
        self._plot_contact = pg.PlotWidget(title="Contact")
        self._plot_cell_grid = pg.PlotWidget(title="Per-Cell Delta-E Grid")

        grid.addWidget(self._plot_de, 0, 0)
        grid.addWidget(self._plot_contrast, 0, 1)
        grid.addWidget(self._plot_energy, 1, 0)
        grid.addWidget(self._plot_homogeneity, 1, 1)
        grid.addWidget(self._plot_contact, 2, 0)
        grid.addWidget(self._plot_cell_grid, 2, 1)

        self._plot_de.setLabel("left", "Delta-E (perceptual units)")
        self._plot_contrast.setLabel("left", "Contrast (a.u.)")
        self._plot_energy.setLabel("left", "Energy (0-1)")
        self._plot_homogeneity.setLabel("left", "Homogeneity (0-1)")
        self._plot_contact.setLabel("left", "Contact (edge count)")

        for plot in [
            self._plot_de,
            self._plot_contrast,
            self._plot_energy,
            self._plot_homogeneity,
            self._plot_contact,
        ]:
            plot.setLabel("bottom", "Time", units="s")
            plot.showGrid(x=True, y=True, alpha=0.3)

        self._plot_cell_grid.hideAxis("left")
        self._plot_cell_grid.hideAxis("bottom")
        self._plot_cell_grid.setAspectLocked(True)
        self._plot_cell_grid.invertY(True)
        self._plot_cell_grid.setMouseEnabled(x=False, y=False)
        self._plot_cell_grid.setMenuEnabled(False)

        self._curve_de = self._plot_de.plot(pen="y", name="Grand Delta-E")
        self._de_extra_curves: List[pg.PlotDataItem] = []

        self._mixing_line = pg.InfiniteLine(
            angle=90,
            pen=pg.mkPen("w", width=2, style=pg.QtCore.Qt.PenStyle.DashLine),
        )
        self._mixing_label = pg.TextItem(color="w", anchor=(0, 1))
        self._mixing_line.setVisible(False)
        self._mixing_label.setVisible(False)
        self._plot_de.addItem(self._mixing_line)
        self._plot_de.addItem(self._mixing_label)

        self._curve_contrast = self._plot_contrast.plot(pen="c")
        self._curve_energy = self._plot_energy.plot(pen="m")
        self._curve_homogeneity = self._plot_homogeneity.plot(pen="g")
        self._curve_contact = self._plot_contact.plot(pen="r")

        self._cell_grid_cmap = pg.colormap.get("viridis")
        self._cell_grid_rects: List[Any] = []
        self._cell_grid_labels: List[pg.TextItem] = []
        self._cell_grid_shape: tuple = (0, 0)
        self._cell_grid_padding: float = 0.12

    def append_data(
        self,
        metrics: Dict[str, Any],
        timestamp: float,
        row_avg: Any = None,
        col_avg: Any = None,
        cell_avg: Any = None,
    ) -> None:
        """Append one frame's metrics to all plots."""
        self._timestamps.append(timestamp)

        for key in self._data:
            if key in metrics:
                self._data[key].append(float(metrics[key]))

        if row_avg is not None:
            self._row_avg_data.append(np.array(row_avg))
        if col_avg is not None:
            self._col_avg_data.append(np.array(col_avg))

        t = np.array(self._timestamps)

        self._curve_contrast.setData(t, np.array(self._data["contrast"]))
        self._curve_energy.setData(t, np.array(self._data["energy"]))
        self._curve_homogeneity.setData(t, np.array(self._data["homogeneity"]))
        self._curve_contact.setData(t, np.array(self._data["contact_perimeter"]))

        if cell_avg is not None:
            self._update_cell_grid(np.asarray(cell_avg, dtype=float))

        self._update_de_plot()

    def _update_cell_grid(self, cell_avg: np.ndarray) -> None:
        """Render per-cell Delta-E as a heatmap grid with numeric labels."""
        rows, cols = self._grid_rows, self._grid_cols
        if cell_avg.size != rows * cols:
            return
        grid = cell_avg.reshape(rows, cols)

        valid = grid[~np.isnan(grid)]
        if valid.size == 0:
            return
        vmin = float(valid.min())
        vmax = float(valid.max())
        denom = vmax - vmin if vmax > vmin else 1.0

        if self._cell_grid_shape != (rows, cols):
            self._rebuild_cell_grid(rows, cols)

        pad = self._cell_grid_padding
        size = 1.0 - 2.0 * pad
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                v = grid[r, c]
                rect = self._cell_grid_rects[idx]
                lbl = self._cell_grid_labels[idx]
                if np.isnan(v):
                    rect.setBrush(pg.mkBrush(60, 60, 60))
                    lbl.setText("—")
                    lbl.setColor("w")
                else:
                    norm = (v - vmin) / denom
                    color = self._cell_grid_cmap.map(norm, mode="qcolor")
                    rect.setBrush(pg.mkBrush(color))
                    lbl.setText(f"{v:.1f}")
                    # Light text on dark cells, dark text on light cells.
                    lbl.setColor("w" if norm < 0.55 else "k")
                rect.setRect(c + pad, r + pad, size, size)
                lbl.setPos(c + 0.5, r + 0.5)

    def _rebuild_cell_grid(self, rows: int, cols: int) -> None:
        """Tear down and recreate the per-cell rect + label items."""
        from PyQt6.QtWidgets import QGraphicsRectItem

        for rect in self._cell_grid_rects:
            self._plot_cell_grid.removeItem(rect)
        for lbl in self._cell_grid_labels:
            self._plot_cell_grid.removeItem(lbl)
        self._cell_grid_rects = []
        self._cell_grid_labels = []

        pen = pg.mkPen(None)
        for r in range(rows):
            for c in range(cols):
                rect = QGraphicsRectItem(0, 0, 1, 1)
                rect.setPen(pen)
                self._plot_cell_grid.addItem(rect)
                self._cell_grid_rects.append(rect)

                lbl = pg.TextItem(anchor=(0.5, 0.5), color="w")
                self._plot_cell_grid.addItem(lbl)
                self._cell_grid_labels.append(lbl)

        self._plot_cell_grid.setXRange(0, cols, padding=0.05)
        self._plot_cell_grid.setYRange(0, rows, padding=0.05)
        self._cell_grid_shape = (rows, cols)

    def _normalize(self, values: np.ndarray) -> np.ndarray:
        """Normalize values to 0-1 range if checkbox is checked."""
        if not self._de_normalize.isChecked():
            return values
        max_val = np.max(values) if len(values) > 0 and np.max(values) > 0 else 1.0
        return values / max_val

    def _update_de_plot(self) -> None:
        t = np.array(self._timestamps)
        mode = self._de_mode.currentIndex()

        # Update Y-axis label based on normalize state
        if self._de_normalize.isChecked():
            self._plot_de.setLabel("left", "Normalized Delta-E (0-1)")
        else:
            self._plot_de.setLabel("left", "Delta-E (perceptual units)")

        for c in self._de_extra_curves:
            self._plot_de.removeItem(c)
        self._de_extra_curves.clear()

        if mode != 0:
            self._mixing_line.setVisible(False)
            self._mixing_label.setVisible(False)

        if mode == 0:
            raw = np.array(self._data["grand_delta_e"])
            self._curve_de.setData(t, self._normalize(raw))
            self._curve_de.setVisible(True)
            self._update_mixing_marker(t, raw)
        elif mode == 1 and self._row_avg_data:
            self._curve_de.setVisible(False)
            n_rows = len(self._row_avg_data[0])
            for r in range(n_rows):
                vals = np.array([d[r] for d in self._row_avg_data if len(d) > r])
                c = self._plot_de.plot(
                    t[: len(vals)],
                    self._normalize(vals),
                    pen=pg.intColor(r, n_rows),
                    name=f"Row {r}",
                )
                self._de_extra_curves.append(c)
        elif mode == 2 and self._col_avg_data:
            self._curve_de.setVisible(False)
            n_cols = len(self._col_avg_data[0])
            for c_idx in range(n_cols):
                vals = np.array([d[c_idx] for d in self._col_avg_data if len(d) > c_idx])
                c = self._plot_de.plot(
                    t[: len(vals)],
                    self._normalize(vals),
                    pen=pg.intColor(c_idx, n_cols),
                    name=f"Col {c_idx}",
                )
                self._de_extra_curves.append(c)

    def _update_mixing_marker(self, t: np.ndarray, raw: np.ndarray) -> None:
        """Mark first time normalized grand Delta-E >= 0.95 (mixing time)."""
        if not self._de_normalize.isChecked() or len(raw) == 0:
            self._mixing_line.setVisible(False)
            self._mixing_label.setVisible(False)
            return
        max_val = float(np.max(raw)) if np.max(raw) > 0 else 1.0
        norm = raw / max_val
        idx = np.argmax(norm >= 0.95)
        if norm[idx] < 0.95:
            self._mixing_line.setVisible(False)
            self._mixing_label.setVisible(False)
            return
        t_mix = float(t[idx])
        self._mixing_line.setPos(t_mix)
        self._mixing_label.setText(f"Mixing time: {t_mix:.2f}s")
        self._mixing_label.setPos(t_mix, 0.95)
        self._mixing_line.setVisible(True)
        self._mixing_label.setVisible(True)

    def _all_metric_plots(self):
        return [
            self._plot_de,
            self._plot_contrast,
            self._plot_energy,
            self._plot_homogeneity,
            self._plot_contact,
        ]

    def set_mixing_result(self, result) -> None:
        """Draw vertical T_mix lines on every metric plot.

        result: MixingTimeResult (duck-typed) or None to clear.
        """
        if not hasattr(self, "_mix_markers"):
            self._mix_markers = []  # list of (plot, item) tuples

        # Remove any existing markers from each plot they were added to.
        for plot, item in self._mix_markers:
            plot.removeItem(item)
        self._mix_markers.clear()

        if result is None:
            return

        main_t_rel = result.t_mix.get(0.95)
        if main_t_rel is None or main_t_rel != main_t_rel:  # NaN guard
            return
        abs_t = result.t_start_s + main_t_rel

        component_times = {
            "ΔE95":      result.t_deltaE.get(0.95),
            "Spatial95": result.t_spatial.get(0.95),
            "Texture95": result.t_texture.get(0.95),
        }

        for plot in self._all_metric_plots():
            line = pg.InfiniteLine(
                pos=abs_t, angle=90,
                pen=pg.mkPen("w", width=2, style=pg.QtCore.Qt.PenStyle.SolidLine),
            )
            plot.addItem(line)
            self._mix_markers.append((plot, line))

            text = pg.TextItem(f"Tmix95 = {main_t_rel:.2f} s", color="w", anchor=(0, 1))
            text.setPos(abs_t, 0)
            plot.addItem(text)
            self._mix_markers.append((plot, text))

            for label, ct in component_times.items():
                if ct is None or ct != ct:
                    continue
                comp_line = pg.InfiniteLine(
                    pos=result.t_start_s + ct, angle=90,
                    pen=pg.mkPen("y", width=1, style=pg.QtCore.Qt.PenStyle.DashLine),
                )
                plot.addItem(comp_line)
                self._mix_markers.append((plot, comp_line))

    def _on_de_mode_changed(self, index: int) -> None:
        self._update_de_plot()

    def clear_data(self) -> None:
        """Reset all plot data."""
        self._timestamps.clear()
        for key in self._data:
            self._data[key].clear()
        self._row_avg_data.clear()
        self._col_avg_data.clear()
        for curve in [
            self._curve_de,
            self._curve_contrast,
            self._curve_energy,
            self._curve_homogeneity,
            self._curve_contact,
        ]:
            curve.setData([], [])
        for lbl in self._cell_grid_labels:
            lbl.setText("")
        for rect in self._cell_grid_rects:
            rect.setBrush(pg.mkBrush(40, 40, 40))
        for c in self._de_extra_curves:
            self._plot_de.removeItem(c)
        self._de_extra_curves.clear()
        self.set_mixing_result(None)

    def set_grid_shape(self, rows: int, cols: int) -> None:
        """Set the spatial grid dimensions for the per-cell Delta-E view."""
        self._grid_rows = rows
        self._grid_cols = cols
        self._rebuild_cell_grid(rows, cols)

    def enable_normalize(self, enabled: bool) -> None:
        """Enable/disable the normalize checkbox. Unchecks when disabled."""
        if not enabled:
            self._de_normalize.setChecked(False)
        self._de_normalize.setEnabled(enabled)

    def save_snapshot(self, path: str) -> None:
        """Save a screenshot of all plots to an image file."""
        pixmap = self.grab()
        pixmap.save(path)
