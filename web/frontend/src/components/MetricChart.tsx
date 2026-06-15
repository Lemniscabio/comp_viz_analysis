import React, { useEffect, useRef } from "react";
import uPlot from "uplot";
import "uplot/dist/uPlot.min.css";

export function MetricChart({ t, y, label }: { t: number[]; y: number[]; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!ref.current) return;
    const opts: uPlot.Options = {
      width: 480, height: 220, title: label,
      scales: { x: { time: false } }, axes: [{ label: "Time (s)" }, {}],
      series: [{}, { label, stroke: "#2563eb", width: 1.5 }],
    };
    const plot = new uPlot(opts, [t, y], ref.current);
    return () => plot.destroy();
  }, [t, y, label]);
  return <div ref={ref} />;
}
