import { useEffect, useRef, useState } from "react";
import { resumeAudio } from "../lib/audioSession.js";
import { Icon, PanelTab } from "./chrome.jsx";
import Spectrum from "./Spectrum.jsx";

const KNOWN_FIELDS = [
  { key: "blade_count", label: "Blades per prop", options: [["", "Any"], ["2", "2"], ["3", "3"], ["4", "4"]] },
  { key: "prop_diameter_inch", label: "Prop size",
    options: [["", "Any"], ["5", "5″"], ["7", "7″"], ["9", "9″"], ["10", "10″"], ["13", "13″"], ["15", "15″"], ["18", "18″"]] },
  { key: "num_motors", label: "Motors", options: [["", "Any"], ["4", "4"], ["6", "6"], ["8", "8"]] },
];
const BADGE = { playing: "PLAYING", recording: "LIVE", done: "DONE" };

function clock(seconds) {
  return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${(seconds % 60).toFixed(1).padStart(4, "0")}`;
}

function useElapsed(elapsed, mode) {
  const [now, setNow] = useState(null);
  useEffect(() => {
    setNow(elapsed());
    if (mode !== "playing" && mode !== "recording") return undefined;
    const timer = setInterval(() => setNow(elapsed()), 100);
    return () => clearInterval(timer);
  }, [elapsed, mode]);
  return now;
}

export default function InputPanel({ session, known, onKnownChange, samples }) {
  const { mode, source, result, error, playSample, playFile, record, stop, elapsed } = session;
  const fileInput = useRef(null);
  const [dragging, setDragging] = useState(false);
  const now = useElapsed(elapsed, mode);
  const live = mode === "playing" || mode === "recording";
  const onFile = (event) => {
    const file = event.target.files?.[0];
    if (file) playFile(file);
    event.target.value = "";
  };
  // WebKit only unlocks audio inside the click itself; the picker's change event comes too late
  const chooseFile = () => {
    resumeAudio();
    fileInput.current?.click();
  };
  useEffect(() => {
    const onDragOver = (event) => event.preventDefault();
    const onDrop = (event) => {
      event.preventDefault();
      setDragging(false);
      const file = event.dataTransfer?.files?.[0];
      if (file) playFile(file);
    };
    window.addEventListener("dragover", onDragOver);
    window.addEventListener("drop", onDrop);
    return () => {
      window.removeEventListener("dragover", onDragOver);
      window.removeEventListener("drop", onDrop);
    };
  }, [playFile]);
  const actions = (
    <>
      <button type="button" className={`btn ${mode === "idle" ? "btn--primary" : ""}`} onClick={chooseFile}>
        <Icon name="upload" />Choose file
      </button>
      <button type="button" className="btn btn--rec" onClick={record}>
        <span className="rec-dot" />{mode === "idle" ? "Record from mic" : "Record"}
      </button>
    </>
  );

  return (
    <section className="panel">
      <PanelTab icon="wave" title="Input" />
      <div className="body">
        <input ref={fileInput} type="file" accept="audio/*,.wav,.mp3,.ogg,.m4a,.flac" hidden onChange={onFile} />
        {mode === "idle" ? (
          <div className={`drop ${dragging ? "over" : ""}`}
            onDragOver={() => setDragging(true)} onDragLeave={() => setDragging(false)}>
            <Icon name="wave" className="big-icon" />
            <h2>Drop a drone recording here</h2>
            <p>WAV, MP3 or OGG. Or record it live from your microphone.</p>
            <div className="actions">{actions}</div>
          </div>
        ) : (
          <>
            <div className="live-head">
              <span className={`live-badge live-badge--${mode}`}><span className="pulse" />{BADGE[mode]}</span>
              <span className="live-src">{source?.label}</span>
              {now != null && (
                <span className="live-time">
                  {source?.duration ? `${clock(Math.min(now, source.duration))} / ${clock(source.duration)}` : clock(now)}
                </span>
              )}
              <div className="live-actions">
                {live && <button type="button" className="btn btn--stop" onClick={stop}><Icon name="stop" />Stop</button>}
                {actions}
              </div>
            </div>
            {mode === "recording" && (
              <p className="mic-note">
                Listening on your mic. Augur assumes a drone is in earshot: fans, AC units and electrical hum
                make the same kind of tone and can fool it.
              </p>
            )}
            {source?.duration ? (
              <div className="progress"><i style={{ width: `${Math.min(1, (now ?? 0) / source.duration) * 100}%` }} /></div>
            ) : null}
            <Spectrum acoustics={result?.acoustics} />
          </>
        )}
        <div className="samples">
          <span className="lbl">Try a sample:</span>
          {samples.map((s) => (
            <button key={s.id} type="button" className="chip-btn" aria-pressed={source?.sampleId === s.id} onClick={() => playSample(s)}>
              {s.label}
            </button>
          ))}
        </div>
        <details className="known">
          <summary>Know anything about it? (optional)</summary>
          <div className="known-grid">
            {KNOWN_FIELDS.map((f) => (
              <label key={f.key}>
                {f.label}
                <select value={known[f.key]} onChange={(e) => onKnownChange(f.key, e.target.value)}>
                  {f.options.map(([value, text]) => <option key={value} value={value}>{text}</option>)}
                </select>
              </label>
            ))}
          </div>
        </details>
        {error && <p className="err">{error}</p>}
      </div>
    </section>
  );
}
