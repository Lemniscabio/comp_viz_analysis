import React, { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import { api, RunStatus, ResultDoc, VideoStatus } from "../lib/api";
import { DeltaEChart } from "../components/DeltaEChart";
import { MetricChart } from "../components/MetricChart";
import { InfoHover } from "../components/InfoHover";
import { DELTA_E_INFO, METRIC_INFO } from "../lib/tooltips";

const CAP = 60;
const fmt = (v: number | null) => (v == null ? "—" : `${v.toFixed(2)} s`);

function VideoResult({ runId, v }: { runId: string; v: VideoStatus }) {
  const [doc, setDoc] = useState<ResultDoc | null>(null);
  useEffect(() => {
    if (v.status !== "done") return;
    (async () => { const { url } = await api.resultUrl(runId, v.video_id); setDoc(await api.fetchResult(url)); })();
  }, [runId, v.video_id, v.status]);
  if (v.status === "failed") return <div><b>{v.filename}</b> — failed: {v.error}</div>;
  if (v.status !== "done" || !doc) return <div><b>{v.filename}</b> — {v.status}…</div>;
  const long = (v.duration_s ?? 0) > CAP, t = doc.series.timestamp;
  return (
    <div style={{ borderTop: "1px solid #eee", padding: "16px 0" }}>
      <h3>{v.filename} <InfoHover info={DELTA_E_INFO} /></h3>
      <DeltaEChart result={doc} />
      <div style={{ margin: "8px 0" }}>
        <b>Mixing time</b>{" "}
        <span style={{ color: "#f59e0b" }}>90%: {fmt(v.t_mix_90_s)}</span>{"  "}
        <span style={{ color: "#10b981" }}>95%: {fmt(v.t_mix_95_s)}</span>{"  "}
        <span style={{ color: "#3b82f6" }}>99%: {fmt(v.t_mix_99_s)}</span>
        {long && <div style={{ color: "#b45309", fontSize: 12, marginTop: 4 }}>
          ℹ Heads up: {v.duration_s?.toFixed(0)}s clip. Mixing-time numbers are most reliable for short clips
          (≤{CAP}s); for long, highly viscous, or dead-zone-prone reactions, sanity-check against the ΔE curve. The graph is valid.
        </div>}
      </div>
      <details>
        <summary style={{ cursor: "pointer" }}>Other metrics (contact, contrast, homogeneity, energy, variance)</summary>
        <div style={{ display: "flex", flexWrap: "wrap", gap: 16, marginTop: 12 }}>
          {Object.values(METRIC_INFO).map((info) => (
            <div key={info.key}>
              <div style={{ fontSize: 13, fontWeight: 600 }}>{info.label}<InfoHover info={info} /></div>
              <MetricChart t={t} y={doc.series[info.key] ?? []} label={info.label} />
            </div>
          ))}
        </div>
      </details>
    </div>
  );
}

export function ResultsView() {
  const { runId } = useParams();
  const [run, setRun] = useState<RunStatus | null>(null);
  useEffect(() => {
    if (!runId) return;
    let alive = true;
    const tick = async () => {
      try { const s = await api.run(runId); if (!alive) return; setRun(s);
        if (s.status !== "done" && s.status !== "failed") setTimeout(tick, 4000); } catch { if (alive) setTimeout(tick, 4000); }
    };
    tick(); return () => { alive = false; };
  }, [runId]);
  if (!run) return <p>Loading…</p>;
  return (
    <div>
      <h2>Run {run.run_id} — {run.status}</h2>
      {run.videos.map((v) => <VideoResult key={v.video_id} runId={run.run_id} v={v} />)}
    </div>
  );
}
