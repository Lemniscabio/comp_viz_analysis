import React, { useEffect, useRef, useState } from "react";
import { BrowserRouter, Routes, Route, Navigate, useLocation } from "react-router-dom";
import { getToken, renderSignIn } from "./lib/auth";
import { api, Me } from "./lib/api";
import { MeContext } from "./lib/me";
import { Spinner } from "./components/Spinner";
import { ProfileHeader } from "./components/ProfileHeader";
import { PendingView } from "./views/PendingView";
import { UploadView } from "./views/UploadView";
import { SelectView } from "./views/SelectView";
import { StatusView } from "./views/StatusView";
import { ResultsView } from "./views/ResultsView";
import { ProfileView } from "./views/ProfileView";

function Pages() {
  const loc = useLocation();
  return (
    <main style={{ maxWidth: 980, margin: "1.5rem auto", padding: "0 16px" }}>
      <div key={loc.pathname} className="kc-page">
        <Routes>
          <Route path="/upload" element={<UploadView />} />
          <Route path="/select" element={<SelectView />} />
          <Route path="/status" element={<StatusView />} />
          <Route path="/runs/:runId" element={<ResultsView />} />
          <Route path="/profile" element={<ProfileView />} />
          <Route path="*" element={<Navigate to="/upload" replace />} />
        </Routes>
      </div>
    </main>
  );
}

export function App() {
  const [signedIn, setSignedIn] = useState(!!getToken());
  const [me, setMe] = useState<Me | null>(null);
  const [loaded, setLoaded] = useState(false);
  const btnRef = useRef<HTMLDivElement>(null);

  useEffect(() => { if (!signedIn && btnRef.current) renderSignIn(btnRef.current, () => setSignedIn(true)); }, [signedIn]);
  useEffect(() => { if (signedIn) api.me().then(setMe).catch(() => setMe(null)).finally(() => setLoaded(true)); }, [signedIn]);

  if (!signedIn)
    return (
      <div className="kc-card" style={{ maxWidth: 420, margin: "4rem auto", padding: 32, textAlign: "center" }}>
        <h1>Kineticolor</h1>
        <p>Sign in with your <b>@lemnisca.bio</b> account.</p>
        <div ref={btnRef} />
      </div>
    );
  if (!loaded) return (
    <div style={{ display: "flex", justifyContent: "center", marginTop: 80 }}>
      <Spinner size={28} />
    </div>
  );

  const active = me?.status === "active";
  return (
    <MeContext.Provider value={me}>
      <BrowserRouter>
        {active ? (
          <>
            <ProfileHeader />
            <Pages />
          </>
        ) : (
          <PendingView />
        )}
      </BrowserRouter>
    </MeContext.Provider>
  );
}
