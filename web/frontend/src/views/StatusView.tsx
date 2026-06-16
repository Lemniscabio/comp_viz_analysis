import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, RunStatus } from "../lib/api";
import { Spinner } from "../components/Spinner";
import { SkeletonRows } from "../components/Skeleton";

function RunBadge({ status }: { status: string }) {
  if (status === "done") return <span className="kc-badge done">● completed</span>;
  if (status === "failed") return <span className="kc-badge fail">▲ failed</span>;
  return <span className="kc-badge run"><Spinner size={12} color="var(--kc-accent)" /> running</span>;
}

export function StatusView() {
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [loading, setLoading] = useState(true);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try { const r = await api.listRuns(); if (alive) setRuns(r.runs); } catch {}
      if (alive) setLoading(false);
      if (alive) setTimeout(tick, 4000);
    };
    tick(); return () => { alive = false; };
  }, []);
  return (
    <div>
      <h2>Runs</h2>
      {loading && (
        <div className="kc-card" style={{ padding: 16 }}>
          <SkeletonRows rows={4} height={40} />
        </div>
      )}
      {!loading && runs.length === 0 && <p>No runs yet.</p>}
      {!loading && runs.length > 0 && (
        <div className="kc-card" style={{ overflow: "hidden" }}>
          <div className="kc-scroll">
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
            <thead><tr style={{ textAlign: "left", color: "var(--kc-muted)" }}>
              <th style={{ padding: "10px 12px" }}>Run</th><th style={{ padding: "10px 12px" }}>Videos</th><th style={{ padding: "10px 12px" }}>Status</th><th style={{ padding: "10px 12px" }}>Done</th><th style={{ padding: "10px 12px" }}></th></tr></thead>
            <tbody className="kc-stagger">
              {runs.map((r) => {
                const done = r.videos.filter((v) => v.status === "done" || v.status === "failed").length;
                return (
                  <tr key={r.run_id} style={{ borderTop: "1px solid var(--kc-border)" }}>
                    <td style={{ padding: "10px 12px" }}>{r.run_id}</td><td style={{ padding: "10px 12px" }}>{r.video_count}</td>
                    <td style={{ padding: "10px 12px" }}><RunBadge status={r.status} /></td>
                    <td style={{ padding: "10px 12px" }}>{done}/{r.video_count}</td>
                    <td style={{ padding: "10px 12px" }}><Link to={`/runs/${r.run_id}`}>view results</Link></td>
                  </tr>
                );
              })}
            </tbody>
          </table>
          </div>
        </div>
      )}
    </div>
  );
}
