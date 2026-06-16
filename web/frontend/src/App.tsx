import React, { useEffect, useRef, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { getToken, renderSignIn } from "./lib/auth";
import { api, Me } from "./lib/api";
import { MeContext } from "./lib/me";
import { ProfileHeader } from "./components/ProfileHeader";
import { PendingView } from "./views/PendingView";
import { UploadView } from "./views/UploadView";
import { SelectView } from "./views/SelectView";
import { StatusView } from "./views/StatusView";
import { ResultsView } from "./views/ResultsView";
import { ProfileView } from "./views/ProfileView";

export function App() {
  const [signedIn, setSignedIn] = useState(!!getToken());
  const [me, setMe] = useState<Me | null>(null);
  const [loaded, setLoaded] = useState(false);
  const btnRef = useRef<HTMLDivElement>(null);

  useEffect(() => { if (!signedIn && btnRef.current) renderSignIn(btnRef.current, () => setSignedIn(true)); }, [signedIn]);
  useEffect(() => { if (signedIn) api.me().then(setMe).catch(() => setMe(null)).finally(() => setLoaded(true)); }, [signedIn]);

  if (!signedIn)
    return (
      <div style={{ maxWidth: 640, margin: "4rem auto", fontFamily: "system-ui" }}>
        <h1>Kineticolor</h1>
        <p>Sign in with your <b>@lemnisca.bio</b> account.</p>
        <div ref={btnRef} />
      </div>
    );
  if (!loaded) return <p style={{ textAlign: "center", marginTop: 80 }}>Loading…</p>;

  const active = me?.status === "active";
  return (
    <MeContext.Provider value={me}>
      <BrowserRouter>
        {active ? (
          <>
            <ProfileHeader />
            <main style={{ maxWidth: 980, margin: "1.5rem auto", fontFamily: "system-ui", padding: "0 16px" }}>
              <Routes>
                <Route path="/upload" element={<UploadView />} />
                <Route path="/select" element={<SelectView />} />
                <Route path="/status" element={<StatusView />} />
                <Route path="/runs/:runId" element={<ResultsView />} />
                <Route path="/profile" element={<ProfileView />} />
                <Route path="*" element={<Navigate to="/upload" replace />} />
              </Routes>
            </main>
          </>
        ) : (
          <PendingView />
        )}
      </BrowserRouter>
    </MeContext.Provider>
  );
}
