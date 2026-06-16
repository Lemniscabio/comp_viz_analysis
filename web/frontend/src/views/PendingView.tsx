import React from "react";
import { useMe } from "../lib/me";

export function PendingView() {
  const me = useMe();
  return (
    <div style={{ maxWidth: 560, margin: "4rem auto", fontFamily: "system-ui", textAlign: "center" }}>
      <h2>Access pending</h2>
      <p>You're signed in as <b>{me?.email}</b> ({me?.status}).</p>
      <p>An administrator needs to grant you access before you can upload or analyze videos.
         Ask <b>kartikey.attri@lemnisca.bio</b> or <b>laalchand.kumawat@lemnisca.bio</b>.</p>
    </div>
  );
}
