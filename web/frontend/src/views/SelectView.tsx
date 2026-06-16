import React, { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, Video } from "../lib/api";
import { Button } from "../components/Button";
import { SkeletonRows } from "../components/Skeleton";

export function SelectView() {
  const [videos, setVideos] = useState<Video[]>([]);
  const [sel, setSel] = useState<Set<string>>(new Set());
  const [busy, setBusy] = useState(false);
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  const nav = useNavigate();

  useEffect(() => {
    api.listVideos().then((r) => setVideos(r.videos)).catch((e) => setErr(String(e))).finally(() => setLoading(false));
  }, []);

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
      {loading && (
        <div className="kc-card" style={{ padding: 16 }}>
          <SkeletonRows rows={4} />
        </div>
      )}
      {!loading && byDate.length === 0 && (
        <div className="kc-card" style={{ padding: 24, textAlign: "center", color: "var(--kc-muted)" }}>
          No uploads yet. Upload some videos first.
        </div>
      )}
      {!loading && byDate.length > 0 && (
        <div className="kc-scroll" style={{ maxHeight: 460, marginBottom: 16 }}>
        <div className="kc-stagger">
          {byDate.map(([date, vids]) => (
            <div key={date} className="kc-card" style={{ padding: "12px 16px", marginBottom: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: "var(--kc-muted)", textTransform: "uppercase", letterSpacing: "0.04em", marginBottom: 6 }}>{date}</div>
              {vids.map((v) => (
                <label key={v.video_id} style={{ display: "flex", alignItems: "center", gap: 10, padding: "8px 0", fontSize: 14 }}>
                  <input type="checkbox" checked={sel.has(v.video_id)} onChange={() => toggle(v.video_id)} />
                  <span>{v.filename} <span style={{ color: "var(--kc-faint)" }}>({(v.size_bytes / 1e6).toFixed(1)} MB)</span></span>
                </label>
              ))}
            </div>
          ))}
        </div>
        </div>
      )}
      <Button loading={busy} disabled={sel.size === 0} onClick={runAnalysis}>
        Run analysis ({sel.size})
      </Button>
    </div>
  );
}
