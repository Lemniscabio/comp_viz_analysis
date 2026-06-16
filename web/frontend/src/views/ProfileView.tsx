import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, ManagedUser, RunStatus, Video } from "../lib/api";
import { useMe, isAdmin } from "../lib/me";
import { Button } from "../components/Button";
import { SkeletonRows } from "../components/Skeleton";

const ROLES = ["admin", "runner", "viewer"];

function UserStatusBadge({ status }: { status: string }) {
  if (status === "active") return <span className="kc-badge done">active</span>;
  if (status === "disabled") return <span className="kc-badge fail">disabled</span>;
  return <span className="kc-badge pending">pending</span>;
}

function AdminUsers() {
  const [users, setUsers] = useState<ManagedUser[]>([]);
  const [roles, setRoles] = useState<Record<string, string>>({});
  const [err, setErr] = useState<string | null>(null);
  async function refresh() {
    const r = await api.listUsers(); setUsers(r.users);
    setRoles(Object.fromEntries(r.users.map((u) => [u.email, u.role ?? "runner"])));
  }
  useEffect(() => { refresh().catch((e) => setErr(String(e))); }, []);
  const sorted = useMemo(() => [...users].sort((a, b) =>
    a.status === "pending" && b.status !== "pending" ? -1 :
    a.status !== "pending" && b.status === "pending" ? 1 : a.email.localeCompare(b.email)), [users]);
  async function decide(email: string, body: { role?: string; status?: string }) {
    try { await api.setUser(email, body); await refresh(); } catch (e) { setErr(String(e)); }
  }
  return (
    <div style={{ marginTop: 24 }}>
      <h3>User management (admin)</h3>
      {err && <p style={{ color: "crimson" }}>{err}</p>}
      <div className="kc-card" style={{ overflow: "hidden" }}>
        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
          <thead><tr style={{ textAlign: "left", color: "var(--kc-muted)" }}>
            <th style={{ padding: "10px 12px" }}>Email</th><th style={{ padding: "10px 12px" }}>Status</th><th style={{ padding: "10px 12px" }}>Role</th><th style={{ padding: "10px 12px" }}>Actions</th></tr></thead>
          <tbody>{sorted.map((u) => (
            <tr key={u.email} style={{ borderTop: "1px solid var(--kc-border)" }}>
              <td style={{ padding: "10px 12px" }}>{u.email}</td><td style={{ padding: "10px 12px" }}><UserStatusBadge status={u.status} /></td>
              <td style={{ padding: "10px 12px" }}><select value={roles[u.email] ?? "runner"}
                    onChange={(e) => setRoles((r) => ({ ...r, [u.email]: e.target.value }))}>
                    {ROLES.map((r) => <option key={r}>{r}</option>)}</select></td>
              <td style={{ padding: "10px 12px" }}>
                <Button variant="primary" onClick={() => decide(u.email, { role: roles[u.email], status: "active" })} style={{ padding: "6px 12px", fontSize: 13 }}>Grant</Button>{" "}
                <Button variant="secondary" onClick={() => decide(u.email, { status: "disabled" })} style={{ padding: "6px 12px", fontSize: 13 }}>Disable</Button>
              </td>
            </tr>))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function ProfileView() {
  const me = useMe();
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  const [runsLoading, setRunsLoading] = useState(true);
  const [videosLoading, setVideosLoading] = useState(true);
  useEffect(() => { api.listRuns().then((r) => setRuns(r.runs)).catch(() => {}).finally(() => setRunsLoading(false)); }, []);
  useEffect(() => { api.myVideos().then((r) => setVideos(r.videos)).catch(() => {}).finally(() => setVideosLoading(false)); }, []);
  return (
    <div>
      <h2>Profile — {me?.email}</h2>
      <section className="kc-card" style={{ padding: 16, marginBottom: 16 }}>
        <h3>My runs</h3>
        {runsLoading ? <SkeletonRows rows={3} /> : runs.length === 0 ? <p>No runs.</p> : (
          <ul>{runs.map((r) => <li key={r.run_id}>
            <Link to={`/runs/${r.run_id}`}>{r.run_id}</Link> — {r.status} ({r.video_count} videos)
          </li>)}</ul>)}
      </section>
      <section className="kc-card" style={{ padding: 16, marginBottom: 16 }}>
        <h3>My uploads</h3>
        {videosLoading ? <SkeletonRows rows={3} /> : videos.length === 0 ? <p>No uploads.</p> : (
          <ul>{videos.map((v) => <li key={v.video_id}>{v.date} — {v.filename}</li>)}</ul>)}
      </section>
      {isAdmin(me) && <AdminUsers />}
    </div>
  );
}
