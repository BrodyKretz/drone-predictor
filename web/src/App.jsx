import { useEffect, useState } from "react";
import { health, reference } from "./api.js";
import EstimatesPanel from "./components/EstimatesPanel.jsx";
import Identification from "./components/Identification.jsx";
import InputPanel from "./components/InputPanel.jsx";
import { TopBar } from "./components/chrome.jsx";
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
  const [ref, setRef] = useState(null);
  const [refError, setRefError] = useState(false);
  const [known, setKnown] = useState(NO_KNOWN);
  const session = useLiveSession(known);

  useEffect(() => {
    health().then(() => setBackendUp(true)).catch(() => setBackendUp(false));
    reference().then(setRef).catch(() => setRefError(true));
  }, []);

  return (
    <div className="app">
      <TopBar backendUp={backendUp} />
      <main className="workspace">
        <div className="col">
          <InputPanel session={session} known={known} samples={SAMPLES}
            onKnownChange={(key, value) => setKnown((k) => ({ ...k, [key]: value }))} />
          <Identification result={session.result} descriptions={ref?.classes} facts={session.resultFacts} />
        </div>
        <EstimatesPanel result={session.result} history={session.history} bands={ref?.bands} facts={session.resultFacts}
          referenceFailed={refError} />
      </main>
    </div>
  );
}
