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
        self._plot_variance = pg.PlotWidget(title="Variance")

        grid.addWidget(self._plot_de, 0, 0)
        grid.addWidget(self._plot_contrast, 0, 1)
        grid.addWidget(self._plot_energy, 1, 0)
        grid.addWidget(self._plot_homogeneity, 1, 1)
        grid.addWidget(self._plot_contact, 2, 0)
        grid.addWidget(self._plot_variance, 2, 1)

        self._plot_de.setLabel("left", "Delta-E (perceptual units)")
        self._plot_contrast.setLabel("left", "Contrast (a.u.)")
        self._plot_energy.setLabel("left", "Energy (0-1)")
        self._plot_homogeneity.setLabel("left", "Homogeneity (0-1)")
        self._plot_contact.setLabel("left", "Contact (edge count)")
        self._plot_variance.setLabel("left", "Variance (a.u.)")

        for plot in [
            self._plot_de,
            self._plot_contrast,
            self._plot_energy,
            self._plot_homogeneity,
            self._plot_contact,
            self._plot_variance,
        ]:
            plot.setLabel("bottom", "Time", units="s")
            plot.showGrid(x=True, y=True, alpha=0.3)

        self._curve_de = self._plot_de.plot(pen="y", name="Grand Delta-E")
        self._de_extra_curves: List[pg.PlotDataItem] = []

        self._curve_contrast = self._plot_contrast.plot(pen="c")
        self._curve_energy = self._plot_energy.plot(pen="m")
        self._curve_homogeneity = self._plot_homogeneity.plot(pen="g")
        self._curve_contact = self._plot_contact.plot(pen="r")

        variance_colors = [
            "r", "g", "b",
            (200, 200, 200), (255, 165, 0), (128, 0, 128), "y",
        ]
        variance_names = ["R", "G", "B", "L*", "a*", "b*", "Delta-E"]
        self._variance_keys = [
            "variance_r", "variance_g", "variance_b",
            "variance_l", "variance_a", "variance_b_star", "variance_delta_e",
        ]
        self._variance_curves = []
        self._plot_variance.addLegend(offset=(10, 10))
        for color, name in zip(variance_colors, variance_names):
            curve = self._plot_variance.plot(
                pen=pg.mkPen(color, width=1), name=name
            )
            self._variance_curves.append(curve)

    def append_data(
        self,
        metrics: Dict[str, Any],
        timestamp: float,
        row_avg: Any = None,
        col_avg: Any = None,
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

        for curve, key in zip(self._variance_curves, self._variance_keys):
            curve.setData(t, np.array(self._data[key]))

        self._update_de_plot()

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

        if mode == 0:
            raw = np.array(self._data["grand_delta_e"])
            self._curve_de.setData(t, self._normalize(raw))
            self._curve_de.setVisible(True)
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
        for curve in self._variance_curves:
            curve.setData([], [])
        for c in self._de_extra_curves:
            self._plot_de.removeItem(c)
        self._de_extra_curves.clear()

    def enable_normalize(self, enabled: bool) -> None:
        """Enable/disable the normalize checkbox. Unchecks when disabled."""
        if not enabled:
            self._de_normalize.setChecked(False)
        self._de_normalize.setEnabled(enabled)

    def save_snapshot(self, path: str) -> None:
        """Save a screenshot of all plots to an image file."""
        pixmap = self.grab()
        pixmap.save(path)
