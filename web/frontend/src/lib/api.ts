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

export interface UploadTarget { video_id: string; filename: string; object_path: string; initiate_url: string; }
export interface Me { email: string; role: string | null; status: string; }
export interface Video { video_id: string; filename: string; date: string; size_bytes: number; owner_email?: string; }
export interface VideoStatus { idx: number; video_id: string; filename: string; status: string;
  duration_s: number | null; t_mix_90_s: number | null; t_mix_95_s: number | null; t_mix_99_s: number | null; error: string | null; }
export interface RunStatus { run_id: string; owner_email: string; status: string; video_count: number; videos: VideoStatus[]; }
export interface ResultDoc { duration_s: number; fps: number; frame_count: number;
  levels: Record<string, number | null>; series: Record<string, number[]>; }
export interface ManagedUser { email: string; role: string | null; status: string; decided_by: string | null; }

export const api = {
  me: () => req<Me>("/api/me"),
  myVideos: () => req<{ videos: Video[] }>("/api/me/videos"),
  myRuns: () => req<{ runs: any[] }>("/api/me/runs"),
  allocate: (files: { name: string; size: number }[]) =>
    req<{ uploads: UploadTarget[] }>("/api/videos:allocate", { method: "POST", body: JSON.stringify({ files }) }),
  finalize: (t: UploadTarget, size: number) =>
    req<Video>(`/api/videos/${t.video_id}:finalize`, { method: "POST",
      body: JSON.stringify({ video_id: t.video_id, filename: t.filename, object_path: t.object_path, size_bytes: size }) }),
  listVideos: () => req<{ videos: Video[] }>("/api/videos"),
  createRun: (video_ids: string[]) => req<RunStatus>("/api/runs", { method: "POST", body: JSON.stringify({ video_ids }) }),
  listRuns: () => req<{ runs: RunStatus[] }>("/api/runs"),
  run: (id: string) => req<RunStatus>(`/api/runs/${id}`),
  resultUrl: (runId: string, videoId: string) => req<{ url: string }>(`/api/runs/${runId}/result/${videoId}`),
  fetchResult: async (url: string) => (await fetch(url)).json() as Promise<ResultDoc>,
  // admin
  listUsers: () => req<{ users: ManagedUser[] }>("/api/admin/users"),
  setUser: (email: string, body: { role?: string; status?: string }) =>
    req<ManagedUser>(`/api/admin/users/${encodeURIComponent(email)}`, { method: "POST", body: JSON.stringify(body) }),
  adminRuns: (user?: string) => req<{ runs: RunStatus[] }>(`/api/admin/runs${user ? `?user=${encodeURIComponent(user)}` : ""}`),
};
