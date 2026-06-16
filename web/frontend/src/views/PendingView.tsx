import React from "react";
import { useMe } from "../lib/me";
import { Spinner } from "../components/Spinner";

export function PendingView() {
  const me = useMe();
  return (
    <div className="kc-card" style={{ maxWidth: 480, margin: "4rem auto", padding: 32, textAlign: "center" }}>
      <Spinner size={24} />
      <h2>Access pending</h2>
      <p>You're signed in as <b>{me?.email}</b> ({me?.status}).</p>
      <p>An administrator needs to grant you access before you can upload or analyze videos.
         Ask <b>kartikey.attri@lemnisca.bio</b>, <b>laalchand.kumawat@lemnisca.bio</b>, or <b>nikhil.bhamwani@lemnisca.bio</b>.</p>
    </div>
  );
}
