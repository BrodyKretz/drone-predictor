import { useEffect, useState } from "react";
import { health } from "./api.js";
import { PanelTab, TopBar } from "./components/chrome.jsx";
import InputPanel from "./components/InputPanel.jsx";
import { useLiveSession } from "./lib/useLiveSession.js";

const SAMPLES = [
  { id: "racing", label: "Racing 5″" },
  { id: "cinematic", label: "Cinematic 9″" },
  { id: "survey", label: "Survey 15″ hex" },
  { id: "military", label: "Military 13″ · 3-blade" },
];
const NO_KNOWN = { blade_count: "", prop_diameter_inch: "", num_motors: "" };

export default function App() {
  const [backendUp, setBackendUp] = useState(null);
  const [known, setKnown] = useState(NO_KNOWN);
  const session = useLiveSession(known);
  const top = session.result?.classes[0];

  useEffect(() => {
    health().then(() => setBackendUp(true)).catch(() => setBackendUp(false));
  }, []);

  return (
    <div className="app">
      <TopBar backendUp={backendUp} />
      <main className="workspace">
        <div className="col">
          <InputPanel session={session} known={known} samples={SAMPLES}
            onKnownChange={(key, value) => setKnown((k) => ({ ...k, [key]: value }))} />
          <section className="panel panel--grow">
            <PanelTab icon="target" title="What it is" />
            <div className="body">{top ? `${top.name} ${Math.round(top.probability * 100)}%` : "Waiting for audio"}</div>
          </section>
        </div>
        <section className="panel"><PanelTab icon="list" title="Live estimates" /></section>
      </main>
    </div>
  );
}
