export async function uploadFile(url: string, file: File, retries = 3): Promise<void> {
  for (let attempt = 0; ; attempt++) {
    try {
      const r = await fetch(url, { method: "PUT", body: file });
      if (!r.ok) throw new Error(`PUT ${r.status}`);
      return;
    } catch (e) {
      if (attempt >= retries) throw e;
      await new Promise((res) => setTimeout(res, 500 * (attempt + 1)));
    }
  }
}

export async function uploadAll(targets: { url: string; file: File }[],
                                concurrency = 6, onProgress?: (done: number) => void): Promise<void> {
  let done = 0, next = 0;
  async function worker() {
    while (next < targets.length) {
      const i = next++;
      await uploadFile(targets[i].url, targets[i].file);
      onProgress?.(++done);
    }
  }
  await Promise.all(Array.from({ length: Math.min(concurrency, targets.length) }, worker));
}
