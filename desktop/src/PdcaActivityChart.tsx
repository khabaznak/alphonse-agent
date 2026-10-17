import { useMemo, useState, type PointerEvent } from "react";
import { PDCA_HISTORY_WINDOW_MS, phaseAt, type PdcaPhase, type PdcaSample } from "./pdcaHistory";

const WIDTH = 248;
const HEIGHT = 96;
const PLOT = { left: 20, right: 6, top: 8, bottom: 16 };
const PHASES: PdcaPhase[] = ["plan", "do", "check", "act"];
const LABELS: Record<PdcaPhase, string> = { plan: "Planning", do: "Doing", check: "Checking", act: "Acting" };

function phaseY(phase: PdcaPhase): number {
  const plotHeight = HEIGHT - PLOT.top - PLOT.bottom;
  return PLOT.top + (PHASES.indexOf(phase) * plotHeight) / (PHASES.length - 1);
}

export function pdcaPaths(samples: PdcaSample[], windowStart: number, now: number): string[] {
  const plotWidth = WIDTH - PLOT.left - PLOT.right;
  const x = (at: number) => PLOT.left + ((Math.max(windowStart, Math.min(now, at)) - windowStart) / PDCA_HISTORY_WINDOW_MS) * plotWidth;
  const changes = samples.filter((sample) => sample.at > windowStart && sample.at <= now);
  let phase = phaseAt(samples, windowStart);
  let startedAt = windowStart;
  let path = phase ? `M ${x(startedAt).toFixed(2)} ${phaseY(phase).toFixed(2)}` : "";
  const paths: string[] = [];

  for (const change of changes) {
    if (phase && path) path += ` H ${x(change.at).toFixed(2)}`;
    if (phase && change.phase && path) {
      path += ` V ${phaseY(change.phase).toFixed(2)}`;
    } else if (phase && !change.phase && path) {
      paths.push(path);
      path = "";
    } else if (!phase && change.phase) {
      path = `M ${x(change.at).toFixed(2)} ${phaseY(change.phase).toFixed(2)}`;
    }
    phase = change.phase;
    startedAt = change.at;
  }
  if (phase && path) paths.push(`${path} H ${x(now).toFixed(2)}`);
  return paths;
}

export function PdcaActivityChart({ samples, now }: { samples: PdcaSample[]; now: number }) {
  const [hoveredAt, setHoveredAt] = useState<number | null>(null);
  const windowStart = now - PDCA_HISTORY_WINDOW_MS;
  const current = phaseAt(samples, now);
  const paths = useMemo(() => pdcaPaths(samples, windowStart, now), [samples, windowStart, now]);
  const hoveredPhase = hoveredAt === null ? null : phaseAt(samples, hoveredAt);

  const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const plotStart = (PLOT.left / WIDTH) * bounds.width;
    const plotWidth = ((WIDTH - PLOT.left - PLOT.right) / WIDTH) * bounds.width;
    const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left - plotStart) / plotWidth));
    setHoveredAt(windowStart + ratio * PDCA_HISTORY_WINDOW_MS);
  };

  return (
    <section className={`sidebar-indicator-panel pdca-activity-panel${current ? " active" : ""}`} aria-label={`PDCA activity: ${current ? LABELS[current] : "Idle"}`}>
      <div className="pdca-activity-heading">
        <span>PDCA</span>
        <strong>{current ? current[0].toUpperCase() : "—"}</strong>
        <small>last 30 min</small>
      </div>
      <div className="queue-chart-wrap">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`PDCA activity chart. Alphonse is ${current ? LABELS[current].toLowerCase() : "idle"}.`} onPointerMove={onPointerMove} onPointerLeave={() => setHoveredAt(null)}>
          {PHASES.map((phase) => <g key={phase}><line className="queue-chart-grid" x1={PLOT.left} x2={WIDTH - PLOT.right} y1={phaseY(phase)} y2={phaseY(phase)} /><text className="queue-chart-label pdca-track-label" x={PLOT.left - 6} y={phaseY(phase) + 2.5} textAnchor="middle">{phase[0].toUpperCase()}</text></g>)}
          <text className="queue-chart-label" x={PLOT.left} y={HEIGHT - 3}>−30m</text>
          <text className="queue-chart-label" x={WIDTH - PLOT.right} y={HEIGHT - 3} textAnchor="end">now</text>
          {paths.map((path, index) => <path className="queue-chart-line pdca-chart-line" d={path} key={`${index}:${path}`} />)}
          {current && <circle className="queue-chart-current" cx={WIDTH - PLOT.right} cy={phaseY(current)} r="3" />}
        </svg>
        {hoveredAt !== null && <div className="queue-chart-tooltip"><strong>{hoveredPhase ? LABELS[hoveredPhase] : "Idle"}</strong><time>{new Date(hoveredAt).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time></div>}
      </div>
    </section>
  );
}
