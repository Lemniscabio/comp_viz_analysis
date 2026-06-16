import React, { useState } from "react";
import { api } from "../lib/api";
import { uploadAll } from "../lib/upload";

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
           style={{ border: `2px dashed ${drag ? "#7c3aed" : "#cbd5e1"}`, borderRadius: 12,
                    padding: 36, textAlign: "center", background: drag ? "#faf5ff" : "#fafafa" }}>
        <p>Drag &amp; drop videos here, or</p>
        <input type="file" multiple accept="video/*" onChange={(e) => add(e.target.files)} />
      </div>
      {files.map((f, i) => (
        <div key={i} style={{ marginTop: 8, fontSize: 13 }}>
          {f.name} — {(f.size / 1e6).toFixed(1)} MB {busy && `(${pct[i] ?? 0}%)`}
        </div>
      ))}
      <button disabled={busy || !files.length} onClick={run} style={{ marginTop: 16 }}>
        {busy ? "Uploading…" : `Upload ${files.length || ""}`}
      </button>
      {msg && <p style={{ marginTop: 10 }}>{msg}</p>}
    </div>
  );
}
