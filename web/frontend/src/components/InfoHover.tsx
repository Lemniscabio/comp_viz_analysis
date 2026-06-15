import React, { useState } from "react";
import type { MetricInfo } from "../lib/tooltips";

export function InfoHover({ info }: { info: MetricInfo }) {
  const [open, setOpen] = useState(false);
  return (
    <span style={{ position: "relative", marginLeft: 6 }}
          onMouseEnter={() => setOpen(true)} onMouseLeave={() => setOpen(false)}>
      <span style={{ cursor: "help", border: "1px solid #888", borderRadius: "50%",
                     fontSize: 11, padding: "0 5px", color: "#555" }}>i</span>
      {open && (
        <div style={{ position: "absolute", zIndex: 10, top: "1.4em", left: 0, width: 300,
                      background: "#fff", border: "1px solid #ccc", borderRadius: 6,
                      padding: 10, boxShadow: "0 4px 12px rgba(0,0,0,.15)", fontSize: 12,
                      lineHeight: 1.4, textAlign: "left", fontWeight: 400 }}>
          <div style={{ marginBottom: 6 }}><b>How it's calculated.</b> {info.how}</div>
          <div><b>How to read it.</b> {info.read}</div>
        </div>
      )}
    </span>
  );
}
