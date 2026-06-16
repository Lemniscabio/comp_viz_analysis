// GCS resumable upload over a signed initiate URL, using XMLHttpRequest.
//
// Why XHR and not fetch: fetch auto-follows the non-standard `308 Resume
// Incomplete` responses GCS returns between chunks and will not reliably expose
// the `Range`/`Location` response headers — which silently stalls the upload at
// 0%. XHR reports the real status code (incl. 308) and headers, and gives true
// per-chunk upload progress.
const CHUNK = 8 * 1024 * 1024; // 8 MiB

interface XhrResult { status: number; range: string | null; location: string | null; }

function xhr(method: string, url: string, headers: Record<string, string>,
             body: Blob | null, onUp?: (loaded: number) => void): Promise<XhrResult> {
  return new Promise((resolve, reject) => {
    const x = new XMLHttpRequest();
    x.open(method, url, true);
    for (const [k, v] of Object.entries(headers)) x.setRequestHeader(k, v);
    if (onUp && body) x.upload.onprogress = (e) => { if (e.lengthComputable) onUp(e.loaded); };
    x.onload = () => resolve({
      status: x.status,
      range: x.getResponseHeader("Range"),
      location: x.getResponseHeader("Location"),
    });
    x.onerror = () => reject(new Error("network error"));
    x.ontimeout = () => reject(new Error("timeout"));
    x.send(body ?? null);
  });
}

async function openSession(initiateUrl: string, contentType: string): Promise<string> {
  // content-type MUST match what the backend signed (application/octet-stream).
  const r = await xhr("POST", initiateUrl,
    { "x-goog-resumable": "start", "content-type": contentType }, null);
  if (r.status < 200 || r.status >= 300) throw new Error(`initiate ${r.status}`);
  if (!r.location) throw new Error("no resumable session location (CORS must expose 'Location')");
  return r.location;
}

function offsetFromRange(range: string | null, fallback: number): number {
  // Range header looks like "bytes=0-8388607"
  if (!range) return fallback;
  const dash = range.lastIndexOf("-");
  return dash >= 0 ? parseInt(range.slice(dash + 1), 10) + 1 : fallback;
}

export async function resumableUpload(
  initiateUrl: string, file: File, onProgress?: (sent: number, total: number) => void,
): Promise<void> {
  const total = file.size;
  const contentType = "application/octet-stream";
  const sessionUri = await openSession(initiateUrl, contentType);
  let offset = 0;
  let consecutiveErrors = 0;
  while (offset < total) {
    const end = Math.min(offset + CHUNK, total);
    const blob = file.slice(offset, end);
    let r: XhrResult;
    try {
      r = await xhr("PUT", sessionUri, { "content-range": `bytes ${offset}-${end - 1}/${total}` },
                    blob, (loaded) => onProgress?.(offset + loaded, total));
    } catch {
      // connection drop — ask GCS how many bytes it actually committed, then retry.
      consecutiveErrors++;
      if (consecutiveErrors > 6) throw new Error("upload failed after repeated drops");
      const q = await xhr("PUT", sessionUri, { "content-range": `bytes */${total}` }, null);
      if (q.status === 200 || q.status === 201) { offset = total; }
      else if (q.status === 308) { offset = offsetFromRange(q.range, offset); }
      onProgress?.(offset, total);
      continue;
    }
    consecutiveErrors = 0;
    if (r.status === 308) {
      offset = offsetFromRange(r.range, end);
    } else if (r.status === 200 || r.status === 201) {
      offset = total;
    } else {
      throw new Error(`chunk PUT ${r.status}`);
    }
    onProgress?.(offset, total);
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
      await resumableUpload(items[i].initiateUrl, items[i].file, (s, t) => onItem?.(i, s, t));
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, items.length) }, worker));
}
