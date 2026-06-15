import React, { useState } from "react";
import { api } from "../lib/api";
import { uploadAll } from "../lib/upload";

export function UploadView({ onSubmitted }: { onSubmitted: (jobId: string) => void }) {
  const [files, setFiles] = useState<File[]>([]);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState<string | null>(null);

  async function run() {
    setBusy(true); setError(null); setProgress(0);
    try {
      const alloc = await api.allocate(files.map((f) => f.name));
      const byName = new Map(files.map((f) => [f.name, f]));
      await uploadAll(alloc.uploads.map((u) => ({ url: u.url, file: byName.get(u.filename)! })),
                      6, setProgress);
      await api.submit(alloc.job_id);
      onSubmitted(alloc.job_id);
    } catch (e) { setError(String(e)); } finally { setBusy(false); }
  }

  return (
    <div>
      <h2>Upload videos</h2>
      <input type="file" multiple accept="video/*"
             onChange={(e) => setFiles(Array.from(e.target.files ?? []))} />
      <p>{files.length} file(s) selected</p>
      <button disabled={busy || files.length === 0} onClick={run}>
        {busy ? `Uploading ${progress}/${files.length}…` : "Analyze"}
      </button>
      {error && <p style={{ color: "crimson" }}>{error}</p>}
    </div>
  );
}
