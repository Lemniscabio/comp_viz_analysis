import React, { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";
import type { ResultDoc } from "../lib/api";

const LEVEL_COLORS: Record<string, string> = { "0.90": "#f59e0b", "0.95": "#10b981", "0.99": "#3b82f6" };

export function DeltaEChart({ result }: { result: ResultDoc }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const t = result.series.timestamp;
    const y = result.series.normalized_delta_e;
    const opts: uPlot.Options = {
      width: 640, height: 320, title: "Normalized ΔE vs time",
      scales: { x: { time: false }, y: { range: [0, 1.05] } },
      axes: [{ label: "Time (s)" }, { label: "Normalized ΔE (0–1)" }],
      series: [{}, { label: "ΔE", stroke: "#7c3aed", width: 2 }],
      hooks: {
        draw: [(u) => {
          // vertical markers at each reached level
          for (const [lvl, tx] of Object.entries(result.levels)) {
            if (tx == null) continue;
            const cx = u.valToPos(tx, "x", true);
            u.ctx.save();
            u.ctx.strokeStyle = LEVEL_COLORS[lvl] ?? "#999";
            u.ctx.setLineDash([4, 3]);
            u.ctx.beginPath();
            u.ctx.moveTo(cx, u.bbox.top);
            u.ctx.lineTo(cx, u.bbox.top + u.bbox.height);
            u.ctx.stroke();
            u.ctx.restore();
          }
        }],
      },
    };
    const plot = new uPlot(opts, [t, y], ref.current);
    return () => plot.destroy();
  }, [result]);
  return <div ref={ref} />;
}
