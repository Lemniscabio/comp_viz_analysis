import React, { useState } from "react";
import { api } from "../lib/api";
import { uploadAll } from "../lib/upload";
import { Button } from "../components/Button";

export function UploadView() {
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [pct, setPct] = useState<number[]>([]);
  const [msg, setMsg] = useState<string | null>(null);
  const [drag, setDrag] = useState(false);

  function add(list: FileList | null) {
    if (list) setFiles((f) => [...f, ...Array.from(list).filter((x) => x.type.startsWith("video/") || /\.(mp4|mov|avi|mkv|m4v)$/i.test(x.name))]);
  }

  async function run() {
    setBusy(true); setMsg(null); setPct(files.map(() => 0));
    try {
      const { uploads } = await api.allocate(files.map((f) => ({ name: f.name, size: f.size })));
      const byName = new Map(files.map((f) => [f.name, f]));
      const items = uploads.map((u) => ({ initiateUrl: u.initiate_url, file: byName.get(u.filename)! }));
      // Serial: one video at a time so each gets the full uplink (faster per-file, stabler).
      await uploadAll(items, 1, (i, sent, total) =>
        setPct((p) => { const n = [...p]; n[i] = Math.round((100 * sent) / total); return n; }));
      for (const u of uploads) await api.finalize(u, byName.get(u.filename)!.size);
      setMsg(`Uploaded ${uploads.length} video(s). Go to Select to analyze.`);
      setFiles([]); setPct([]);
    } catch (e) { setMsg(String(e)); } finally { setBusy(false); }
  }

  return (
    <div>
      <h2>Upload videos</h2>
      <div onDragOver={(e) => { e.preventDefault(); setDrag(true); }}
           onDragLeave={() => setDrag(false)}
           onDrop={(e) => { e.preventDefault(); setDrag(false); add(e.dataTransfer.files); }}
           style={{ border: `2px dashed ${drag ? "var(--kc-accent)" : "var(--kc-border)"}`, borderRadius: "var(--kc-radius)",
                    padding: 36, textAlign: "center", background: drag ? "var(--kc-accent-weak)" : "var(--kc-surface)",
                    transition: "border-color 150ms ease, background-color 150ms ease" }}>
        <p>Drag &amp; drop videos here, or</p>
        <input type="file" multiple accept="video/*" onChange={(e) => add(e.target.files)} />
      </div>
      {files.length > 0 && (
        <div className="kc-card" style={{ padding: "4px 16px", marginTop: 16 }}>
          {files.map((f, i) => (
            <div key={i} style={{ padding: "10px 0" }}>
              <div style={{ display: "flex", justifyContent: "space-between", fontSize: 13, marginBottom: 6 }}>
                <span>{f.name}</span>
                <span style={{ color: "var(--kc-muted)" }}>
                  {(f.size / 1e6).toFixed(1)} MB {busy && `· ${pct[i] ?? 0}%`}
                </span>
              </div>
              {busy && <div className="kc-progress"><span style={{ width: `${pct[i] ?? 0}%` }} /></div>}
            </div>
          ))}
        </div>
      )}
      <Button loading={busy} disabled={!files.length} onClick={run} style={{ marginTop: 16 }}>
        {busy ? "Uploading…" : `Upload ${files.length || ""}`}
      </Button>
      {msg && <p style={{ marginTop: 10 }}>{msg}</p>}
    </div>
  );
}
