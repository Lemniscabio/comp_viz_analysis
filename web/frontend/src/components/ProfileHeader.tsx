import React from "react";
import { Link, useLocation } from "react-router-dom";
import { useMe, isAdmin } from "../lib/me";
import { clearToken } from "../lib/auth";
import { Button } from "./Button";

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
                     borderBottom: "1px solid var(--kc-border)", position: "sticky", top: 0,
                     background: "rgba(255,255,255,0.8)", backdropFilter: "blur(8px)", zIndex: 20 }}>
      <strong style={{ fontSize: 16 }}>Kineticolor</strong>
      <nav style={{ display: "flex", gap: 12, flex: 1 }}>
        {tabs.map((t) => {
          const active = loc.pathname.startsWith(t.to);
          return (
            <Link key={t.to} to={t.to} style={{
              textDecoration: "none", fontSize: 14, fontWeight: active ? 650 : 500,
              color: active ? "var(--kc-accent)" : "var(--kc-muted)",
              background: active ? "var(--kc-accent-weak)" : "transparent",
              padding: "6px 12px", borderRadius: 999, transition: "background-color 150ms ease, color 150ms ease",
            }}>{t.label}</Link>
          );
        })}
      </nav>
      <span style={{ fontSize: 13, color: "#444" }}>
        {me?.email} {isAdmin(me) && <em style={{ color: "#7c3aed" }}>(admin)</em>}
      </span>
      <Button variant="ghost" onClick={() => { clearToken(); location.reload(); }}>Sign out</Button>
    </header>
  );
}
