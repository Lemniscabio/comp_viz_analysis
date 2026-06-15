import React, { useEffect, useRef, useState } from "react";
import { getToken, renderSignIn } from "./lib/auth";
import { UploadView } from "./views/UploadView";
import { ResultView } from "./views/ResultView";

export function App() {
  const [signedIn, setSignedIn] = useState(!!getToken());
  const [jobId, setJobId] = useState<string | null>(null);
  const btnRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!signedIn && btnRef.current) renderSignIn(btnRef.current, () => setSignedIn(true));
  }, [signedIn]);

  if (!signedIn) {
    return (
      <div style={{ maxWidth: 640, margin: "4rem auto", fontFamily: "system-ui" }}>
        <h1>Kineticolor</h1>
        <p>Sign in with your <b>@lemnisca.bio</b> account.</p>
        <div ref={btnRef} />
      </div>
    );
  }
  return (
    <div style={{ maxWidth: 900, margin: "2rem auto", fontFamily: "system-ui" }}>
      <h1>Kineticolor — Mixing-Time Analysis</h1>
      <UploadView onSubmitted={setJobId} />
      {jobId && <ResultView jobId={jobId} />}
    </div>
  );
}
