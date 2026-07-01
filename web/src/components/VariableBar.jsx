import { CONFIDENCE_COLORS, fmt, prettyName, relativeWidth } from "../lib/format.js";

// One variable row. The band width encodes RELATIVE uncertainty (interval width
// over |median|) — a narrow band means a confident estimate. Color encodes the
// confidence label. This mirrors the CLI's "confidence ladder" visually.
export default function VariableBar({ varKey, summary, attribution }) {
  const rw = relativeWidth(summary);
  const color = CONFIDENCE_COLORS[summary.confidence] || "#94a3b8";

  // Map relative width to a band that fills 0–100% of the track (clamped at 1.5).
  const bandPct = Math.min(rw, 1.5) / 1.5 * 100;
  const unit = summary.unit ? ` ${summary.unit}` : "";

  return (
    <div className="var-row">
      <div className="var-head">
        <span className="var-name">{prettyName(varKey)}</span>
        <span className="var-median">
          {fmt(summary.median)}
          <span className="var-unit">{unit}</span>
        </span>
      </div>

      <div className="var-track" title={`relative width ${(rw * 100).toFixed(0)}%`}>
        <div
          className="var-band"
          style={{ width: `${Math.max(bandPct, 3)}%`, background: color }}
        />
      </div>

      <div className="var-foot">
        <span className="var-interval">
          90%: {fmt(summary.interval_low)}–{fmt(summary.interval_high)}
        </span>
        <span className="var-conf" style={{ color }}>
          {summary.confidence}
        </span>
        {attribution && attribution.gain > 0.05 && (
          <span className="var-attr">
            ← {attribution.source} (−{(attribution.gain * 100).toFixed(0)}%)
          </span>
        )}
      </div>
    </div>
  );
}
