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
        <div className="kc-scroll">
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
    </div>
  );
}

function AdminRuns() {
  const [runs, setRuns] = useState<RunStatus[]>([]);
  const [owner, setOwner] = useState<string>("");
  const [loading, setLoading] = useState(true);
  const [err, setErr] = useState<string | null>(null);
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try { const r = await api.adminRuns(); if (alive) setRuns(r.runs); }
      catch (e) { if (alive) setErr(String(e)); }
      if (alive) setLoading(false);
      if (alive) setTimeout(tick, 6000);
    };
    tick(); return () => { alive = false; };
  }, []);
  const owners = useMemo(() => [...new Set(runs.map((r) => r.owner_email))].sort(), [runs]);
  const shown = useMemo(() => owner ? runs.filter((r) => r.owner_email === owner) : runs, [runs, owner]);
  return (
    <div style={{ marginTop: 24 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, marginBottom: 8 }}>
        <h3 style={{ margin: 0 }}>All runs (admin)</h3>
        <select value={owner} onChange={(e) => setOwner(e.target.value)} style={{ marginLeft: "auto" }}>
          <option value="">All users ({runs.length})</option>
          {owners.map((o) => <option key={o} value={o}>{o}</option>)}
        </select>
      </div>
      {err && <p style={{ color: "crimson" }}>{err}</p>}
      <div className="kc-card" style={{ overflow: "hidden" }}>
        <div className="kc-scroll">
          {loading ? <div style={{ padding: 16 }}><SkeletonRows rows={4} /></div> :
           shown.length === 0 ? <p style={{ padding: 16, color: "var(--kc-muted)" }}>No runs.</p> : (
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 14 }}>
              <thead><tr style={{ textAlign: "left", color: "var(--kc-muted)" }}>
                <th style={{ padding: "10px 12px" }}>Run</th><th style={{ padding: "10px 12px" }}>Owner</th>
                <th style={{ padding: "10px 12px" }}>Videos</th><th style={{ padding: "10px 12px" }}>Status</th>
                <th style={{ padding: "10px 12px" }}>Done</th><th style={{ padding: "10px 12px" }}></th></tr></thead>
              <tbody>
                {shown.map((r) => {
                  const done = r.videos.filter((v) => v.status === "done" || v.status === "failed").length;
                  return (
                    <tr key={r.run_id} style={{ borderTop: "1px solid var(--kc-border)" }}>
                      <td style={{ padding: "10px 12px" }}>{r.run_id}</td>
                      <td style={{ padding: "10px 12px" }}>{r.owner_email}</td>
                      <td style={{ padding: "10px 12px" }}>{r.video_count}</td>
                      <td style={{ padding: "10px 12px" }}>
                        <span className={`kc-badge ${r.status === "done" ? "done" : r.status === "failed" ? "fail" : "run"}`}>{r.status}</span></td>
                      <td style={{ padding: "10px 12px" }}>{done}/{r.video_count}</td>
                      <td style={{ padding: "10px 12px" }}><Link to={`/runs/${r.run_id}`}>view results</Link></td>
                    </tr>
                  );
                })}
              </tbody>
            </table>)}
        </div>
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
  const uploadsByDate = useMemo(() => {
    const m = new Map<string, Video[]>();
    for (const v of videos) { if (!m.has(v.date)) m.set(v.date, []); m.get(v.date)!.push(v); }
    return [...m.entries()].sort((a, b) => b[0].localeCompare(a[0]));
  }, [videos]);
  return (
    <div>
      <h2>Profile — {me?.email}</h2>
      <section className="kc-card" style={{ padding: 16, marginBottom: 16 }}>
        <h3>My runs</h3>
        {runsLoading ? <SkeletonRows rows={3} /> : runs.length === 0 ? <p style={{ color: "var(--kc-muted)" }}>No runs.</p> : (
          <div className="kc-scroll">
            {runs.map((r) => (
              <div key={r.run_id} className="kc-row" style={{ display: "flex", alignItems: "center", gap: 8, padding: "8px", fontSize: 14 }}>
                <Link to={`/runs/${r.run_id}`} style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{r.run_id}</Link>
                <span className={`kc-badge ${r.status === "done" ? "done" : r.status === "failed" ? "fail" : "run"}`}>{r.status}</span>
                <span style={{ color: "var(--kc-faint)", fontSize: 12 }}>{r.video_count} videos</span>
              </div>
            ))}
          </div>)}
      </section>
      <section className="kc-card" style={{ padding: 16, marginBottom: 16 }}>
        <h3>My uploads</h3>
        {videosLoading ? <SkeletonRows rows={3} /> : videos.length === 0 ? <p style={{ color: "var(--kc-muted)" }}>No uploads.</p> : (
          <div className="kc-scroll">
            {uploadsByDate.map(([date, vids]) => (
              <div key={date} style={{ marginBottom: 10 }}>
                <div style={{ fontSize: 12, fontWeight: 600, color: "var(--kc-faint)", textTransform: "uppercase", letterSpacing: ".04em", margin: "6px 4px" }}>{date} · {vids.length}</div>
                {vids.map((v) => (
                  <div key={v.video_id} className="kc-row" style={{ display: "flex", alignItems: "center", gap: 8, padding: "6px 8px", fontSize: 14 }}>
                    <span style={{ color: "var(--kc-faint)" }}>🎞</span>
                    <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>{v.filename}</span>
                    <span style={{ color: "var(--kc-faint)", fontSize: 12 }}>{(v.size_bytes / 1e6).toFixed(1)} MB</span>
                  </div>
                ))}
              </div>
            ))}
          </div>)}
      </section>
      {isAdmin(me) && <AdminRuns />}
      {isAdmin(me) && <AdminUsers />}
    </div>
  );
}
