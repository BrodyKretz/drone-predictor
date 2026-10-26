const W = 96;
const H = 20;

export default function Sparkline({ points, factor }) {
  if (points.length < 2) return <svg className="spark" aria-hidden="true" />;
  const lows = points.map((p) => p.low * factor);
  const highs = points.map((p) => p.high * factor);
  const medians = points.map((p) => p.median * factor);
  const lo = Math.min(...lows);
  const hi = Math.max(...highs);
  const useLog = lo > 0 && hi / lo > 20;
  const tr = (v) => (useLog ? Math.log10(v) : v);
  const a = tr(lo);
  const b = tr(hi) === a ? a + 1 : tr(hi);
  const x = (i) => (i / (points.length - 1)) * (W - 2) + 1;
  const y = (v) => H - 1 - ((tr(v) - a) / (b - a)) * (H - 2);
  const path = (values) => values.map((v, i) => `${x(i).toFixed(1)},${y(v).toFixed(1)}`);
  const top = path(highs);
  const bottom = path(lows).reverse();
  return (
    <svg className="spark" viewBox={`0 0 ${W} ${H}`} aria-hidden="true">
      <path className="band" d={`M${top.join("L")}L${bottom.join("L")}Z`} />
      <path className="line" d={`M${path(medians).join("L")}`} />
    </svg>
  );
}
