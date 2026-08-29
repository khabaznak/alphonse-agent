import { useMemo, useState, type PointerEvent } from "react";
import { QUEUE_HISTORY_WINDOW_MS, queueWorkload, type QueueSample } from "./queueHistory";

const WIDTH = 248;
const HEIGHT = 78;
const PLOT = { left: 20, right: 6, top: 7, bottom: 16 };

function closestSample(samples: QueueSample[], target: number): QueueSample | null {
  return samples.reduce<QueueSample | null>((closest, sample) => (
    !closest || Math.abs(sample.at - target) < Math.abs(closest.at - target) ? sample : closest
  ), null);
}

export function QueueWorkloadChart({ samples }: { samples: QueueSample[] }) {
  const [hovered, setHovered] = useState<QueueSample | null>(null);
  const latest = samples.at(-1) || null;
  const latestAt = latest?.at || Date.now();
  const windowStart = latestAt - QUEUE_HISTORY_WINDOW_MS;
  const total = latest ? queueWorkload(latest) : 0;
  const chart = useMemo(() => {
    const visible = samples.filter((sample) => sample.at >= windowStart);
    const maximum = Math.max(1, ...visible.map(queueWorkload));
    const plotWidth = WIDTH - PLOT.left - PLOT.right;
    const plotHeight = HEIGHT - PLOT.top - PLOT.bottom;
    const point = (sample: QueueSample) => ({
      x: PLOT.left + ((sample.at - windowStart) / QUEUE_HISTORY_WINDOW_MS) * plotWidth,
      y: PLOT.top + (1 - queueWorkload(sample) / maximum) * plotHeight,
    });
    const points = visible.map(point);
    const path = points.reduce((value, current, index) => {
      if (index === 0) return `M ${current.x.toFixed(2)} ${current.y.toFixed(2)}`;
      return `${value} H ${current.x.toFixed(2)} V ${current.y.toFixed(2)}`;
    }, "");
    return { maximum, path, lastPoint: points.at(-1) || null };
  }, [samples, windowStart]);

  const onPointerMove = (event: PointerEvent<SVGSVGElement>) => {
    if (!samples.length) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const plotStart = (PLOT.left / WIDTH) * bounds.width;
    const plotWidth = ((WIDTH - PLOT.left - PLOT.right) / WIDTH) * bounds.width;
    const ratio = Math.max(0, Math.min(1, (event.clientX - bounds.left - plotStart) / plotWidth));
    setHovered(closestSample(samples, windowStart + ratio * QUEUE_HISTORY_WINDOW_MS));
  };

  return (
    <section className={`queue-workload-panel${total > 0 ? " active" : ""}`} aria-label={`Queue workload: ${total} total tasks`}>
      <div className="queue-workload-heading">
        <span>Workload</span>
        <strong>{total}</strong>
        <small>last 30 min</small>
      </div>
      <div className="queue-chart-wrap">
        <svg viewBox={`0 0 ${WIDTH} ${HEIGHT}`} role="img" aria-label={`Queue workload chart. ${total} total tasks: ${latest?.ready || 0} waiting and ${latest?.processing || 0} processing.`} onPointerMove={onPointerMove} onPointerLeave={() => setHovered(null)}>
          <line className="queue-chart-axis" x1={PLOT.left} x2={WIDTH - PLOT.right} y1={HEIGHT - PLOT.bottom} y2={HEIGHT - PLOT.bottom} />
          {[0, .5, 1].map((ratio) => <line className="queue-chart-grid" key={ratio} x1={PLOT.left} x2={WIDTH - PLOT.right} y1={PLOT.top + ratio * (HEIGHT - PLOT.top - PLOT.bottom)} y2={PLOT.top + ratio * (HEIGHT - PLOT.top - PLOT.bottom)} />)}
          <text className="queue-chart-label" x={PLOT.left - 4} y={PLOT.top + 4} textAnchor="end">{chart.maximum}</text>
          <text className="queue-chart-label" x={PLOT.left - 4} y={HEIGHT - PLOT.bottom + 3} textAnchor="end">0</text>
          <text className="queue-chart-label" x={PLOT.left} y={HEIGHT - 3}>−30m</text>
          <text className="queue-chart-label" x={WIDTH - PLOT.right} y={HEIGHT - 3} textAnchor="end">now</text>
          {chart.path && <path className="queue-chart-line" d={chart.path} />}
          {chart.lastPoint && <circle className="queue-chart-current" cx={chart.lastPoint.x} cy={chart.lastPoint.y} r="3" />}
        </svg>
        {hovered && <div className="queue-chart-tooltip"><strong>{queueWorkload(hovered)} total</strong><span>{hovered.ready} waiting · {hovered.processing} working</span><time>{new Date(hovered.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</time></div>}
      </div>
    </section>
  );
}
