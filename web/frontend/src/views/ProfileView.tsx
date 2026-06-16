import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { api, ManagedUser, RunStatus, Video } from "../lib/api";
import { useMe, isAdmin } from "../lib/me";

const ROLES = ["admin", "runner", "viewer"];

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
      <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
        <thead><tr style={{ textAlign: "left", color: "#6b7280" }}>
          <th>Email</th><th>Status</th><th>Role</th><th>Actions</th></tr></thead>
        <tbody>{sorted.map((u) => (
          <tr key={u.email} style={{ borderTop: "1px solid #eee" }}>
            <td>{u.email}</td><td>{u.status}</td>
            <td><select value={roles[u.email] ?? "runner"}
                  onChange={(e) => setRoles((r) => ({ ...r, [u.email]: e.target.value }))}>
                  {ROLES.map((r) => <option key={r}>{r}</option>)}</select></td>
            <td>
              <button onClick={() => decide(u.email, { role: roles[u.email], status: "active" })}>Grant</button>{" "}
              <button onClick={() => decide(u.email, { status: "disabled" })}>Disable</button>
            </td>
          </tr>))}
        </tbody>
      </table>
    </div>
  );
}

export function ProfileView() {
  const me = useMe();
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [videos, setVideos] = useState<Video[]>([]);
  useEffect(() => { api.listRuns().then((r) => setRuns(r.runs)).catch(() => {}); }, []);
  useEffect(() => { api.myVideos().then((r) => setVideos(r.videos)).catch(() => {}); }, []);
  return (
    <div>
      <h2>Profile — {me?.email}</h2>
      <h3>My runs</h3>
      {runs.length === 0 ? <p>No runs.</p> : (
        <ul>{runs.map((r) => <li key={r.run_id}>
          <Link to={`/runs/${r.run_id}`}>{r.run_id}</Link> — {r.status} ({r.video_count} videos)
        </li>)}</ul>)}
      <h3>My uploads</h3>
      {videos.length === 0 ? <p>No uploads.</p> : (
        <ul>{videos.map((v) => <li key={v.video_id}>{v.date} — {v.filename}</li>)}</ul>)}
      {isAdmin(me) && <AdminUsers />}
    </div>
  );
}
