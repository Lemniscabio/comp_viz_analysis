import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, Video } from "../lib/api";

export function SelectView() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const nav = useNavigate();

  useEffect(() => { api.listVideos().then((r) => setVideos(r.videos)).catch((e) => setErr(String(e))); }, []);

  const byDate = useMemo(() => {
    const m = new Map<string, Video[]>();
    for (const v of videos) { if (!m.has(v.date)) m.set(v.date, []); m.get(v.date)!.push(v); }
    return [...m.entries()].sort((a, b) => b[0].localeCompare(a[0]));
  }, [videos]);

  function toggle(id: string) {
    setSel((s) => { const n = new Set(s); n.has(id) ? n.delete(id) : n.add(id); return n; });
  }

  async function runAnalysis() {
    setBusy(true); setErr(null);
    try {
      const run = await api.createRun([...sel]);
      nav("/status");
      return run;
    } catch (e) { setErr(String(e)); } finally { setBusy(false); }
  }

  return (
    <div>
      <h2>Select videos to analyze</h2>
      {err && <p style={{ color: "crimson" }}>{err}</p>}
      {byDate.length === 0 && <p>No uploads yet. Upload some videos first.</p>}
      {byDate.map(([date, vids]) => (
        <div key={date} style={{ marginBottom: 16 }}>
          <div style={{ fontWeight: 700, color: "#374151", margin: "8px 0" }}>{date}</div>
          {vids.map((v) => (
            <label key={v.video_id} style={{ display: "block", padding: "4px 0", fontSize: 14 }}>
              <input type="checkbox" checked={sel.has(v.video_id)} onChange={() => toggle(v.video_id)} />{" "}
              {v.filename} <span style={{ color: "#9ca3af" }}>({(v.size_bytes / 1e6).toFixed(1)} MB)</span>
            </label>
          ))}
        </div>
      ))}
      <button disabled={busy || sel.size === 0} onClick={runAnalysis}>
        {busy ? "Starting…" : `Run analysis (${sel.size})`}
      </button>
    </div>
  );
}
