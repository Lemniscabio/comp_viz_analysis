// GCS resumable upload over a signed initiate URL. Survives connection drops by
// resuming from the last byte the server acknowledges.
const CHUNK = 8 * 1024 * 1024; // 8 MiB

async function openSession(initiateUrl: string, contentType: string): Promise<string> {
  const r = await fetch(initiateUrl, {
    method: "POST",
    headers: { "x-goog-resumable": "start", "content-type": contentType },
  });
  if (!r.ok) throw new Error(`initiate ${r.status}`);
  const loc = r.headers.get("location");
  if (!loc) throw new Error("no resumable session location (CORS must expose 'location')");
  return loc;
}

async function committedOffset(sessionUri: string, total: number): Promise<number> {
  // Query current offset: PUT with empty body + Content-Range: bytes */total
  const r = await fetch(sessionUri, {
    method: "PUT",
    headers: { "content-range": `bytes */${total}` },
  });
  if (r.status === 200 || r.status === 201) return total; // already done
  if (r.status === 308) {
    const range = r.headers.get("range"); // e.g. "bytes=0-8388607"
    if (!range) return 0;
    return parseInt(range.split("-")[1], 10) + 1;
  }
  throw new Error(`status query ${r.status}`);
}

export async function resumableUpload(
  initiateUrl: string, file: File, onProgress?: (sent: number, total: number) => void,
): Promise<void> {
  const total = file.size;
  const contentType = file.type || "application/octet-stream";
  let sessionUri = await openSession(initiateUrl, contentType);
  let offset = 0;
  while (offset < total) {
    const end = Math.min(offset + CHUNK, total);
    const blob = file.slice(offset, end);
    try {
      const r = await fetch(sessionUri, {
        method: "PUT",
        headers: { "content-range": `bytes ${offset}-${end - 1}/${total}` },
        body: blob,
      });
      if (r.status === 308) {
        const range = r.headers.get("range");
        offset = range ? parseInt(range.split("-")[1], 10) + 1 : end;
      } else if (r.status === 200 || r.status === 201) {
        offset = total;
      } else {
        throw new Error(`chunk ${r.status}`);
      }
      onProgress?.(offset, total);
    } catch (e) {
      // connection drop — resync offset from server and retry this chunk
      offset = await committedOffset(sessionUri, total);
      onProgress?.(offset, total);
    }
  }
}

export async function uploadAll(
  items: { initiateUrl: string; file: File }[],
  concurrency = 3,
  onItem?: (i: number, sent: number, total: number) => void,
): Promise<void> {
  let next = 0;
  async function worker() {
    while (next < items.length) {
      const i = next++;
      await resumableUpload(items[i].initiateUrl, items[i].file,
                            (s, t) => onItem?.(i, s, t));
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
}
