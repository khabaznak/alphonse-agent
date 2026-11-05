import { useEffect, useRef, useState } from "react";
import { daemonRequest } from "./api";
import type { WorkItem } from "./WorkOverview";

export type CoordinationItem = { id: string; title: string; status: "pending" | "completed" | "cancelled"; when?: string };
export type CoordinationData = { family: "todos" | "shopping" | "reminders" | "appointments"; artifactId: string; owner: string; source: string; updatedAt: string; expiresAt: number; items: CoordinationItem[] };
export function validCoordination(data: CoordinationData | undefined): data is CoordinationData {
  return !!data && ["todos", "shopping", "reminders", "appointments"].includes(data.family) && typeof data.artifactId === "string" && data.artifactId.startsWith("artifact.") && typeof data.owner === "string" && typeof data.source === "string" && typeof data.updatedAt === "string" && Number.isFinite(Date.parse(data.updatedAt)) && Number.isFinite(data.expiresAt) && Array.isArray(data.items) && data.items.length <= 50 && new Set(data.items.map((item) => item?.id)).size === data.items.length && data.items.every((item) => item && typeof item.id === "string" && item.id.length > 0 && item.id.length <= 128 && typeof item.title === "string" && item.title.length > 0 && item.title.length <= 300 && ["pending", "completed", "cancelled"].includes(item.status) && (item.when === undefined || typeof item.when === "string" && Number.isFinite(Date.parse(item.when))));
}
export type CoordinationRoutes = { onAgenda: (id?: string) => void; onWork?: (project: string, taskId?: string) => void; onItem?: (artifactId: string, itemId: string, family: CoordinationData["family"]) => void; onRequest?: (text: string) => void; timezone?: string };
export function widgetTime(value: string, timezone = "UTC") {
  return new Date(value).toLocaleString(undefined, { timeZone: timezone });
}
/** Creates a draft only. Sending, authorization, execution and verification use the ordinary skill workflow. */
export function appendWidgetDraft(current: string, draft: string) {
  return current.includes(draft) ? current : current.trim() ? `${current}\n\n${draft}` : draft;
}
export function workflowDraft(data: CoordinationData, item: CoordinationItem | undefined, action: string) {
  return `Use the owning skill workflow for ${data.family}. Requested action: ${action}.\nResource: ${data.artifactId}; owner: ${data.owner}; source revision: ${data.updatedAt}.\n${item ? `Item ID: ${JSON.stringify(item.id)}; displayed title: ${JSON.stringify(item.title)}.` : "List-level edit."}\nRe-read the current source, check authorization and prerequisites, and verify the effect before reporting success. Refresh the widget snapshot after a confirmed change.`;
}
export function CoordinationList({ data, onRequest, onAgenda, onWork, onItem, timezone }: CoordinationRoutes & { data: CoordinationData }) {
  if (!validCoordination(data) || data.expiresAt <= Date.now()) return <p role="status">Unavailable. The owning skill needs to refresh this source.</p>;
  return <div className="coordination-list">
    {!data.items.length && <p>No items in this list.</p>}
    <ul>{data.items.map((item) => <li key={item.id} className={`coordination-row coordination-${item.status}`}><div className="coordination-row-title"><span className="coordination-mark" aria-hidden="true">{item.status === "completed" ? "✓" : ""}</span><strong>{item.title}</strong></div><small>{item.status}{item.when && <> · {widgetTime(item.when, timezone)}</>}</small><details className="widget-row-details"><summary>Details &amp; actions</summary><small>Item: {item.id}</small>
      {onItem && <button onClick={() => onItem(data.artifactId, item.id, data.family)}>Open in {data.family === "appointments" || data.family === "reminders" ? "Agenda" : "Projects & Tasks"}</button>}{onRequest && <div className="coordination-actions">{item.status === "pending" && ["todos", "shopping"].includes(data.family) && <button onClick={() => onRequest(workflowDraft(data, item, "Complete this item"))}>Request completion</button>}<button onClick={() => onRequest(workflowDraft(data, item, data.family === "appointments" ? "Review or reschedule this appointment" : "Edit this item"))}>Request edit</button></div>}</details>
    </li>)}</ul>
    {onRequest && <button onClick={() => onRequest(workflowDraft(data, undefined, "Add or edit list items"))}>Add / edit with Alphonse</button>}
    <details className="widget-source-details"><summary>Source &amp; details</summary><small>{data.source} · Owner: {data.owner}</small><small>Source updated: {widgetTime(data.updatedAt, timezone)}</small><p>Requests open a conversation draft. Items change here after the source confirms the effect.</p></details>
    {data.family === "appointments" || data.family === "reminders" ? <button onClick={() => onAgenda()}>Open Agenda</button> : <button onClick={() => onWork?.("")}>Projects &amp; Tasks</button>}
  </div>;
}
export type Reminder = { scheduled_task_id: string; owner_user_id: string; project_id: string; name: string; status: string; delivery_mode: string; next_run_at: string | null };
export type NativeCoordination = { member: string; loading: boolean; error: string; work: WorkItem[]; reminders: Reminder[] };
export function useNativeCoordination(member: string, enabled: boolean) {
  const [state, setState] = useState<NativeCoordination>({ member: "", loading: true, error: "", work: [], reminders: [] });
  const [notice, setNotice] = useState("");
  const [busy, setBusy] = useState("");
  const generation = useRef(0);
  const inflight = useRef(false);
  const current = useRef({ member, enabled }); current.current = { member, enabled };
  async function load() {
    const id = ++generation.current;
    if (!member || !enabled) return;
    const [work, timed] = await Promise.allSettled([
      daemonRequest<{ items: WorkItem[] }>("work_items", { actor_user_id: member, owner_user_id: member, view: "active", limit: 8 }),
      daemonRequest<{ tasks: Reminder[] }>("scheduled_tasks", { actor_user_id: member, owner_user_id: member, limit: 100 }),
    ]);
    if (id !== generation.current || current.current.member !== member || !current.current.enabled) return;
    setState({ member, loading: false, error: [work.status === "rejected" ? "Tasks unavailable." : "", timed.status === "rejected" ? "Reminders unavailable." : ""].join(" "), work: work.status === "fulfilled" ? work.value.items : [], reminders: timed.status === "fulfilled" ? timed.value.tasks.filter((task) => task.delivery_mode === "direct" && ["active", "paused"].includes(task.status)) : [] });
  }
  useEffect(() => {
    setState({ member, loading: true, error: "", work: [], reminders: [] }); setNotice("");
    void load();
    const timer = window.setInterval(() => { if (!inflight.current) void load(); }, 5000);
    return () => { generation.current++; window.clearInterval(timer); };
  }, [member, enabled]);
  async function change(task: Reminder, action: "pause" | "resume" | "cancel") {
    if (inflight.current || !enabled || !member || task.owner_user_id !== member) return;
    inflight.current = true; generation.current++; setBusy(task.scheduled_task_id); setNotice("");
    try {
      const result = await daemonRequest<{ task: Reminder }>(`${action}_schedule`, { actor_user_id: member, scheduled_task_id: task.scheduled_task_id });
      const expected = { pause: "paused", resume: "active", cancel: "cancelled" }[action];
      if (result.task?.scheduled_task_id !== task.scheduled_task_id || result.task.owner_user_id !== member || result.task.status !== expected) throw new Error("The expected change was not confirmed.");
      if (current.current.member === member && current.current.enabled) {
        setNotice(`Confirmed: ${expected}.`);
        setState((old) => ({ ...old, reminders: old.reminders.map((row) => row.scheduled_task_id === task.scheduled_task_id ? result.task : row).filter((row) => row.status !== "cancelled") }));
        await load();
      }
    } catch (cause) {
      if (current.current.member === member) setNotice(`Save failed or unconfirmed: ${cause instanceof Error ? cause.message : String(cause)}. Refresh before retrying.`);
    } finally { inflight.current = false; setBusy(""); }
  }
  return { state: state.member === member && enabled ? state : { member, loading: true, error: "", work: [], reminders: [] }, notice, busy, change, refresh: load };
}
export function NativeCoordinationBody({ family, controller, onWork, onAgenda, onRequest, timezone }: CoordinationRoutes & { family: string; controller: ReturnType<typeof useNativeCoordination> }) {
  const { state, busy, notice, change } = controller;
  if (state.loading) return <p role="status">Loading {family}…</p>;
  const error = family === "todos" ? state.error.includes("Tasks unavailable") : state.error.includes("Reminders unavailable");
  if (error) return <><p role="alert">Unavailable. Check access and reconnect.</p><button onClick={() => void controller.refresh()}>Retry</button></>;
  if (family === "todos") return <div className="coordination-list">{!state.work.length && <p>No active to-dos.</p>}<ul>{state.work.map((item) => <li key={item.task_id}><strong>{item.goal}</strong><small>{item.status} · {item.next_action}</small><small>{item.project_name || "Everyday work"}</small><details className="widget-row-details"><summary>Details &amp; actions</summary><small>Task: {item.task_id}</small><button onClick={() => onWork?.(item.project_id, item.task_id)}>Open Projects &amp; Tasks</button>{onRequest && <button onClick={() => onRequest(`Review task ${JSON.stringify(item.task_id)} in project ${JSON.stringify(item.project_id)}: ${JSON.stringify(item.goal)}. Check current state and acceptance evidence before making changes.`)}>Review with Alphonse</button>}</details></li>)}</ul><details className="widget-source-details"><summary>Source &amp; details</summary><small>Persisted delegated work · Owner: {state.member}</small><p>Execution records determine completion. Editable personal lists come from their owning skill.</p></details><button onClick={() => onWork?.("")}>All tasks</button></div>;
  return <div className="coordination-list"><small>Native reminder schedules · Owner: {state.member}</small>{notice && <p role={notice.startsWith("Confirmed:") ? "status" : "alert"}>{notice}</p>}{busy && <p role="status">Saving reminder…</p>}{!state.reminders.length && <p>No active or paused reminders.</p>}<ul>{state.reminders.map((task) => <li key={task.scheduled_task_id}><strong>{task.name}</strong><small>{task.status} · {task.next_run_at ? widgetTime(task.next_run_at, timezone) : "No next run"}</small><small>Schedule: {task.scheduled_task_id}</small><div className="coordination-actions"><button disabled={!!busy} onClick={() => void change(task, task.status === "paused" ? "resume" : "pause")}>{task.status === "paused" ? "Resume" : "Pause"}</button><button disabled={!!busy} onClick={() => { if (window.confirm(`Cancel reminder “${task.name}”?`)) void change(task, "cancel"); }}>Cancel</button><button onClick={() => onAgenda(task.scheduled_task_id)}>Details / edit</button></div></li>)}</ul><button disabled={!!busy} onClick={() => void controller.refresh()}>Refresh</button><p>Showing up to 100 stored schedules. Open Agenda for all schedule controls and history.</p></div>;
}
