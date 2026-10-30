import { widgetTime } from "./CoordinationWidgets";

type Base = { id: string; title: string; scope: "session" | "project" | "habit" | "sample"; recorded_at: string };
export type ProgressRecord = Base & { status: "active" | "completed" | "cancelled"; value: number; target: number; unit: string; display: "bar" | "dots" };
export type TimerRecord = Base & { status: "running" | "paused" | "completed" | "cancelled"; elapsed_ms: number; duration_ms: number; running_since: string | null };
export type ActivityData = { family: "progress" | "timer" | "tracker"; artifactId: string; owner: string; source: string; updatedAt: string; expiresAt: number; items: (ProgressRecord | TimerRecord)[] };
const timestamp = (value: unknown): value is string => typeof value === "string" && /(?:Z|[+-]\d{2}:\d{2})$/.test(value) && Number.isFinite(Date.parse(value));
const number = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value) && value >= 0 && value <= 1e12;
const text = (value: unknown, limit: number): value is string => typeof value === "string" && value.length > 0 && value.length <= limit;

export function validActivity(data: ActivityData | undefined): data is ActivityData {
  if (!data || !["progress", "timer", "tracker"].includes(data.family) || !text(data.artifactId, 256) || !data.artifactId.startsWith("artifact.") || !text(data.owner, 256) || !text(data.source, 300) || !timestamp(data.updatedAt) || !Number.isFinite(data.expiresAt) || data.expiresAt <= Date.parse(data.updatedAt) || !Array.isArray(data.items) || data.items.length > 50) return false;
  const ids = new Set<string>();
  return data.items.every((item) => {
    if (!item || !text(item.id, 128) || ids.has(item.id) || !text(item.title, 300) || !["session", "project", "habit", "sample"].includes(item.scope) || !timestamp(item.recorded_at) || Date.parse(item.recorded_at) > Date.parse(data.updatedAt)) return false;
    ids.add(item.id);
    if (data.family === "timer") {
      const timer = item as TimerRecord;
      return ["running", "paused", "completed", "cancelled"].includes(timer.status) && number(timer.elapsed_ms) && number(timer.duration_ms) && timer.duration_ms > 0 && (timer.status === "running" ? timestamp(timer.running_since) && Date.parse(timer.running_since) <= Date.parse(data.updatedAt) : timer.running_since === null);
    }
    const progress = item as ProgressRecord;
    return ["active", "completed", "cancelled"].includes(progress.status) && number(progress.value) && number(progress.target) && progress.target > 0 && text(progress.unit, 40) && ["bar", "dots"].includes(progress.display) && (progress.display !== "dots" || progress.target <= 100 && Number.isInteger(progress.target) && Number.isInteger(progress.value));
  });
}

/** Derive each frame from source time. Never increment counters or mutate source state. */
export function timerFrame(timer: TimerRecord, now: number) {
  const elapsed = timer.elapsed_ms + (timer.status === "running" && timer.running_since ? Math.max(0, now - Date.parse(timer.running_since)) : 0);
  return { elapsed, remaining: Math.max(0, timer.duration_ms - elapsed), fraction: Math.min(1, elapsed / timer.duration_ms), expired: elapsed >= timer.duration_ms };
}
export function duration(ms: number) {
  const seconds = Math.ceil(ms / 1000);
  return `${Math.floor(seconds / 60)} min ${seconds % 60} sec`;
}
const scopeLabel = { session: "Recorded session", project: "Project progress", habit: "Habit / activity record", sample: "Illustrative sample · not recorded progress" };

export function ActivityWidget({ data, now, timezone, onRequest }: { data: ActivityData; now: number; timezone?: string; onRequest?: (draft: string) => void }) {
  if (!validActivity(data) || data.expiresAt <= now) return <p role="status">Unavailable. The owning skill needs to refresh this source.</p>;
  function request(item: ProgressRecord | TimerRecord, action: string) {
    onRequest?.(`Use the owning skill workflow to ${action}. Resource: ${JSON.stringify(data.artifactId)}; item: ${JSON.stringify(item.id)}; source revision: ${JSON.stringify(data.updatedAt)}. Re-read the source, check authorization and prerequisites, persist the timestamps and verify the effect before reporting success. Refresh the activity snapshot.`);
  }
  return <div className="activity-widget"><small>{data.source} · Owner: {data.owner}</small><small>Source updated: {widgetTime(data.updatedAt, timezone)}</small>
    {!data.items.length && <p>No recorded activity.</p>}
    {data.items.map((item) => <section key={item.id} aria-label={item.title}><h4>{item.title}</h4><small>{scopeLabel[item.scope]} · {widgetTime(item.recorded_at, timezone)}</small>
      {data.family === "timer" ? (() => {
        const timer = item as TimerRecord;
        const frame = timerFrame(timer, now);
        const label = timer.status === "completed" ? "Source reports completed" : timer.status === "cancelled" ? "Cancelled" : frame.expired ? "Time elapsed · completion unconfirmed" : timer.status === "paused" ? "Paused" : "Running";
        return <><div className="visual-time" style={{ background: `conic-gradient(var(--accent) ${frame.fraction * 360}deg, var(--border) 0deg)` }} aria-hidden="true"><span /></div><p role="timer" aria-label={`${item.title}: ${duration(frame.remaining)} remaining; ${label}`}>{duration(frame.remaining)} remaining</p><small>{duration(frame.elapsed)} elapsed · {duration(timer.duration_ms)} planned</small><p>{label}</p>{onRequest && ["running", "paused"].includes(timer.status) && <button onClick={() => request(timer, timer.status === "paused" ? "resume this timer" : "pause this timer")}>Request {timer.status === "paused" ? "resume" : "pause"}</button>}</>;
      })() : (() => {
        const progress = item as ProgressRecord;
        const label = `${progress.value} of ${progress.target} ${progress.unit}`;
        return <><p>{label}</p><div role="progressbar" aria-label={item.title} aria-valuemin={0} aria-valuemax={progress.target} aria-valuenow={Math.min(progress.value, progress.target)} aria-valuetext={label} className={`recorded-progress ${progress.display}`}>
          {progress.display === "dots" ? Array.from({ length: progress.target }, (_, index) => <i key={index} data-filled={index < progress.value} aria-hidden="true" />) : <span style={{ width: `${Math.min(100, progress.value / progress.target * 100)}%` }} />}
        </div><small>Source status: {progress.status}</small></>;
      })()}
    </section>)}
    <p>Records remain authoritative. Requests open a conversation draft; timer expiry does not confirm task completion.</p>
  </div>;
}
