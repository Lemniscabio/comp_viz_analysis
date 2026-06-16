import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, RunStatus } from "../lib/api";

export function StatusView() {
  const [runs, setRuns] = useState<RunStatus[]>([]);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try { const r = await api.listRuns(); if (alive) setRuns(r.runs); } catch {}
      if (alive) setTimeout(tick, 4000);
    };
    tick(); return () => { alive = false; };
  }, []);
  return (
    <div>
      <h2>Runs</h2>
      {runs.length === 0 && <p>No runs yet.</p>}
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
        <thead><tr style={{ textAlign: "left", color: "#6b7280" }}>
          <th>Run</th><th>Videos</th><th>Status</th><th>Done</th><th></th></tr></thead>
        <tbody>
          {runs.map((r) => {
            const done = r.videos.filter((v) => v.status === "done" || v.status === "failed").length;
            return (
              <tr key={r.run_id} style={{ borderTop: "1px solid #eee" }}>
                <td>{r.run_id}</td><td>{r.video_count}</td>
                <td>{r.status === "done" ? "✅ completed" : r.status === "failed" ? "⚠ failed" : "⏳ running"}</td>
                <td>{done}/{r.video_count}</td>
                <td><Link to={`/runs/${r.run_id}`}>view results</Link></td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
