import React from "react";
import { Link, useLocation } from "react-router-dom";
import { useMe, isAdmin } from "../lib/me";
import { clearToken } from "../lib/auth";

const tabs = [
  { to: "/upload", label: "Upload" },
  { to: "/select", label: "Select" },
  { to: "/status", label: "Status" },
  { to: "/profile", label: "Profile" },
];

export function ProfileHeader() {
  const me = useMe();
  const loc = useLocation();
  return (
    <header style={{ display: "flex", alignItems: "center", gap: 16, padding: "12px 20px",
                     borderBottom: "1px solid #e5e7eb", position: "sticky", top: 0, background: "#fff", zIndex: 20 }}>
      <strong style={{ fontSize: 16 }}>Kineticolor</strong>
      <nav style={{ display: "flex", gap: 12, flex: 1 }}>
        {tabs.map((t) => (
          <Link key={t.to} to={t.to} style={{ textDecoration: "none",
            fontWeight: loc.pathname.startsWith(t.to) ? 700 : 400,
            color: loc.pathname.startsWith(t.to) ? "#111" : "#666" }}>{t.label}</Link>
        ))}
      </nav>
      <span style={{ fontSize: 13, color: "#444" }}>
        {me?.email} {isAdmin(me) && <em style={{ color: "#7c3aed" }}>(admin)</em>}
      </span>
      <button onClick={() => { clearToken(); location.reload(); }}>Sign out</button>
    </header>
  );
}
