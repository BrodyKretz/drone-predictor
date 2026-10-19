import { useLayoutEffect, useRef, useState } from "react";

const HEIGHT = 170;
const M = { left: 40, right: 12, top: 24, bottom: 22 };
const FLOOR_DB = -80;

function useWidth(ref) {
  const [width, setWidth] = useState(0);
  useLayoutEffect(() => {
    if (!ref.current) return undefined;
    const observer = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    observer.observe(ref.current);
    return () => observer.disconnect();
  }, [ref]);
  return width;
}

function niceStep(span, count) {
  const raw = span / Math.max(1, count);
  const p = 10 ** Math.floor(Math.log10(raw));
  const m = raw / p;
  return (m < 1.5 ? 1 : m < 3 ? 2 : m < 7 ? 5 : 10) * p;
}

export default function Spectrum({ acoustics }) {
  const host = useRef(null);
  const width = useWidth(host);
  const ready = acoustics && acoustics.spectrum_freq_hz.length > 0 && width > 0;
  return (
    <div className="spectrum" ref={host}>
      {ready ? <Plot acoustics={acoustics} width={width} /> : <div className="wait">Listening…</div>}
    </div>
  );
}

function Plot({ acoustics, width }) {
  const freqs = acoustics.spectrum_freq_hz;
  const db = acoustics.spectrum_db;
  const fmax = freqs[freqs.length - 1];
  const x = (f) => M.left + (f / fmax) * (width - M.left - M.right);
  const y = (d) => M.top + (-(Number.isFinite(d) ? d : FLOOR_DB) / 80) * (HEIGHT - M.top - M.bottom);
  const points = freqs.map((f, i) => `${x(f).toFixed(1)},${y(db[i]).toFixed(1)}`).join(" L");
  const step = niceStep(fmax, Math.max(3, Math.floor((width - M.left - M.right) / 90)));
  const ticks = [];
  for (let t = 0; t <= fmax + 1e-9; t += step) ticks.push(t);
  const nearestDb = (f) => {
    let j = 0;
    for (let i = 1; i < freqs.length; i++) if (Math.abs(freqs[i] - f) < Math.abs(freqs[j] - f)) j = i;
    return db[j];
  };
  return (
    <svg width={width} height={HEIGHT} viewBox={`0 0 ${width} ${HEIGHT}`} role="img" aria-label="Live audio spectrum">
      <defs>
        <linearGradient id="spectrum-fill" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#a986ec" stopOpacity="0.3" />
          <stop offset="1" stopColor="#a986ec" stopOpacity="0" />
        </linearGradient>
      </defs>
      {[0, -40, -80].map((d) => (
        <g key={d}>
          <line className="grid" x1={M.left} x2={width - M.right} y1={y(d)} y2={y(d)} />
          <text className="ax-tick" x={M.left - 6} y={y(d) + 3.5} textAnchor="end">{d === 0 ? "0 dB" : `−${-d}`}</text>
        </g>
      ))}
      <path d={`M${x(freqs[0])},${y(-80)} L${points} L${x(fmax)},${y(-80)}Z`} fill="url(#spectrum-fill)" />
      <path className="spec-line" d={`M${points}`} />
      <line className="ax-line" x1={M.left} x2={width - M.right} y1={HEIGHT - M.bottom} y2={HEIGHT - M.bottom} />
      {ticks.map((t) => (
        <text key={t} className="ax-tick" x={x(t)} y={HEIGHT - M.bottom + 13} textAnchor="middle">
          {t >= 1000 ? Math.round(t).toLocaleString("en-US") : Math.round(t)}{t === 0 ? " Hz" : ""}
        </text>
      ))}
      {acoustics.harmonics.map((h) => {
        const label = h.order === 1 ? "BPF" : `${h.order}×`;
        if (h.matched_hz == null) {
          const hx = x(h.expected_hz);
          return (
            <g key={h.order}>
              <line className="h-miss" x1={hx} x2={hx} y1={M.top - 8} y2={HEIGHT - M.bottom} />
              <text className="h-miss-label" x={hx} y={M.top - 12} textAnchor="middle">{label}?</text>
            </g>
          );
        }
        const hx = x(h.matched_hz);
        const hy = y(nearestDb(h.matched_hz));
        return (
          <g key={h.order}>
            <line className="h-hit" x1={hx} x2={hx} y1={M.top - 8} y2={Math.max(hy - 8, M.top - 8)} />
            <path className="h-hit-mark" d={`M${hx} ${hy - 2}l-4 -6h8z`} />
            <text className="h-hit-label" x={hx} y={M.top - 12} textAnchor="middle">{label}</text>
          </g>
        );
      })}
    </svg>
  );
}
