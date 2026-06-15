import { getToken, clearToken } from "./auth";

async function req<T>(path: string, init: RequestInit = {}): Promise<T> {
  const headers = new Headers(init.headers);
  const token = getToken();
  if (token) headers.set("Authorization", `Bearer ${token}`);
  headers.set("Content-Type", "application/json");
  const r = await fetch(path, { ...init, headers });
  if (r.status === 401) { clearToken(); throw new Error("session expired"); }
  if (!r.ok) throw new Error(`${r.status}: ${await r.text()}`);
  return r.json() as Promise<T>;
}

export interface UploadTarget { idx: number; filename: string; object_path: string; url: string; }
export interface AllocateResp { job_id: string; uploads: UploadTarget[]; }
export interface VideoStatus {
  idx: number; filename: string; status: string; duration_s: number | null;
  t_mix_90_s: number | null; t_mix_95_s: number | null; t_mix_99_s: number | null; error: string | null;
}
export interface JobStatus { job_id: string; status: string; video_count: number; videos: VideoStatus[]; }
export interface ResultDoc {
  duration_s: number; fps: number; frame_count: number;
  levels: Record<string, number | null>;
  series: Record<string, number[]>;
}

export const api = {
  allocate: (files: string[]) => req<AllocateResp>("/api/jobs:allocate",
    { method: "POST", body: JSON.stringify({ files }) }),
  submit: (job_id: string) => req<JobStatus>("/api/jobs:submit",
    { method: "POST", body: JSON.stringify({ job_id }) }),
  status: (job_id: string) => req<JobStatus>(`/api/jobs/${job_id}`),
  resultUrl: (job_id: string, idx: number) => req<{ url: string }>(`/api/jobs/${job_id}/result/${idx}`),
  fetchResult: async (signedUrl: string) => (await fetch(signedUrl)).json() as Promise<ResultDoc>,
};
