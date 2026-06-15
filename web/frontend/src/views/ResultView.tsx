import React, { useEffect, useState } from "react";
import { api, JobStatus, ResultDoc, VideoStatus } from "../lib/api";
import { DeltaEChart } from "../components/DeltaEChart";
import { MetricChart } from "../components/MetricChart";
import { InfoHover } from "../components/InfoHover";
import { DELTA_E_INFO, METRIC_INFO } from "../lib/tooltips";

const DURATION_CAP_S = 60;
const fmt = (v: number | null) => (v == null ? "—" : `${v.toFixed(2)} s`);

function VideoResult({ jobId, v }: { jobId: string; v: VideoStatus }) {
  const [doc, setDoc] = useState<ResultDoc | null>(null);
  useEffect(() => {
    if (v.status !== "done") return;
    (async () => {
      const { url } = await api.resultUrl(jobId, v.idx);
      setDoc(await api.fetchResult(url));
    })();
  }, [jobId, v.idx, v.status]);

  if (v.status === "failed") return <div><b>{v.filename}</b> — failed: {v.error}</div>;
  if (v.status !== "done" || !doc) return <div><b>{v.filename}</b> — {v.status}…</div>;

  const longVideo = (v.duration_s ?? 0) > DURATION_CAP_S;
  const t = doc.series.timestamp;

  return (
    <div style={{ borderTop: "1px solid #eee", padding: "16px 0" }}>
      <h3>{v.filename} <InfoHover info={DELTA_E_INFO} /></h3>
      <DeltaEChart result={doc} />
      <div style={{ margin: "8px 0" }}>
        <b>Mixing time</b>{" "}
        <span style={{ color: "#f59e0b" }}>90%: {fmt(v.t_mix_90_s)}</span>{"  "}
        <span style={{ color: "#10b981" }}>95%: {fmt(v.t_mix_95_s)}</span>{"  "}
        <span style={{ color: "#3b82f6" }}>99%: {fmt(v.t_mix_99_s)}</span>
        {longVideo && (
          <div style={{ color: "#b45309", fontSize: 12, marginTop: 4 }}>
            ℹ Heads up: this clip is {v.duration_s?.toFixed(0)}s. Mixing-time numbers are most
            reliable for short clips (≤{DURATION_CAP_S}s); for long, highly viscous, or
            dead-zone-prone reactions, sanity-check them against the ΔE curve. The graph itself is valid.
          </div>
        )}
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

export function ResultView({ jobId }: { jobId: string }) {
  const [job, setJob] = useState<JobStatus | null>(null);
  useEffect(() => {
    let alive = true;
    async function tick() {
      try {
        const s = await api.status(jobId);
        if (!alive) return;
        setJob(s);
        if (s.status !== "done" && s.status !== "failed") setTimeout(tick, 4000);
      } catch { if (alive) setTimeout(tick, 4000); }
    }
    tick();
    return () => { alive = false; };
  }, [jobId]);

  if (!job) return <p>Loading…</p>;
  return (
    <div>
      <h2>Job {job.job_id} — {job.status}</h2>
      {job.videos.map((v) => <VideoResult key={v.idx} jobId={job.job_id} v={v} />)}
    </div>
  );
}
