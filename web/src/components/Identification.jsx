import { confidenceWord, fmtPercent } from "../lib/format.js";
import { PanelTab } from "./chrome.jsx";

const CLASS_NAMES = {
  racing: "Racing / FPV quad",
  cinematic: "Cinematic camera drone",
  survey: "Survey / mapping drone",
  military: "Military / ISR drone",
};
const displayName = (name) => CLASS_NAMES[name] || name;

const SPECIAL = {
  no_tone: () => ["Listening…", "No steady drone tone yet. Keep the drone in earshot, ideally hovering."],
  not_hovering: (r) => ["Can't tell yet", `The pitch is changing (${r.acoustics.flight_state.state}), so the hover physics behind the guess doesn't apply. Wait for a steady hover.`],
  no_class_fits: () => ["Unknown type", "None of the four known classes fits these inputs."],
};

function whyLines(result) {
  const ac = result.acoustics;
  const fs = ac.flight_state;
  const lines = [];
  if (ac.fundamental_hz) {
    const found = ac.harmonics.filter((h) => h.matched_hz != null).length;
    lines.push(<>Blade-pass tone <b>{ac.fundamental_hz.toFixed(0)} Hz</b> with <b>{found}/{ac.harmonics.length}</b> harmonics found</>);
  }
  if (fs.state === "hover" && fs.pitch_variation != null) {
    lines.push(<>Hovering: pitch steady (variation {(fs.pitch_variation * 100).toFixed(1)}% ≤ {(fs.steady_threshold * 100).toFixed(0)}%)</>);
  }
  if (result.blades.length > 1) {
    lines.push(
      <>
        Blades per prop:{" "}
        {result.blades.map((b, i) => (
          <span key={b.blades}>{i ? " or " : ""}<b>{b.blades}</b> ({fmtPercent(b.probability)})</span>
        ))}
        . Not given, so both are considered.
      </>,
    );
  }
  return lines;
}

function Guess({ result, descriptions, known }) {
  const top = result.classes[0];
  const ok = result.status === "ok";
  const [specialName, specialSub] = ok ? [] : SPECIAL[result.status](result);
  const warnings = [...result.notes];
  if (ok && top.probability < 0.6) {
    warnings.push(
      known.prop_diameter_inch
        ? "Several classes overlap even with the prop size you gave; the guess can't narrow further from audio."
        : "Several classes sound alike from audio alone. Telling it the prop size (above) can narrow it down.",
    );
  }
  return (
    <>
      <div className="guess">
        {ok ? (
          <>
            <span className="guess-name">{displayName(top.name)}</span>
            <span className="guess-p">{fmtPercent(top.probability)}</span>
          </>
        ) : (
          <span className="guess-name dim">{specialName}</span>
        )}
      </div>
      <div className="guess-sub">
        {ok ? `${confidenceWord(top.probability)} · from ${result.duration_s.toFixed(1)} s of audio` : specialSub}
      </div>
      {ok && descriptions?.[top.name] && <div className="guess-desc">{descriptions[top.name]}</div>}
      <div className="bars">
        {result.classes.map((c, i) => {
          const isTop = ok && i === 0;
          return (
            <div className="bar-row" key={c.name}>
              <span className={`name ${isTop ? "top" : ""}`}>{displayName(c.name)}</span>
              <div className={`bar ${isTop ? "top" : ""}`}><i style={{ width: `${(c.probability * 100).toFixed(1)}%` }} /></div>
              <span className="pct">{fmtPercent(c.probability)}</span>
            </div>
          );
        })}
      </div>
      <div className="why">
        {whyLines(result).map((line, i) => <div key={i}>{line}</div>)}
        {warnings.map((w) => <div key={w} className="warn">{w}</div>)}
      </div>
    </>
  );
}

export default function Identification({ result, descriptions, known }) {
  return (
    <section className="panel panel--grow">
      <PanelTab icon="target" title="What it is" />
      <div className="body body--scroll">
        <div className="eyebrow">Best guess</div>
        {result ? (
          <Guess result={result} descriptions={descriptions} known={known} />
        ) : (
          <>
            <div className="guess"><span className="guess-name dim">Waiting for audio</span></div>
            <div className="guess-sub">Drop a file, record from the mic, or try a sample above.</div>
          </>
        )}
      </div>
    </section>
  );
}
