import { useCallback, useEffect, useRef, useState } from "react";
import { daemonRequest } from "./api";
import type { Project } from "./types";

export type WorkItem = {
  task_id: string; owner_user_id: string; project_id: string; project_name: string; everyday: boolean;
  goal: string; status: string; blocked: boolean; next_action: string; completed_criteria: number;
  criteria: Array<{ id: string; statement: string; status: string }>;
  waiting_for_user_id?: string | null; can_answer?: boolean;
  created_at: string; updated_at: string; conversation_day: string | null;
};
type WorkPage = { project_summaries?: Record<string, { total: number; completed: number; failed: number; cancelled: number }>; items: WorkItem[]; has_more: boolean; limit: number; offset: number };
type Schedule = { scheduled_task_id: string; project_id: string; owner_user_id: string; name: string; status: string; next_run_at: string | null; latest_execution?: { status: string } | null };
export type WorkView = "active" | "attention" | "history";
export const workStatus = (status: string) => ({ done: "Completed", waiting_user: "Waiting for input", queued: "Queued", running: "Running", failed: "Failed", cancelled: "Cancelled", active: "Active", paused: "Paused", completed: "Completed", archived: "Archived" }[status] || status);
export function scheduleMatches(task: Schedule, view: WorkView) {
  if (view === "history") return ["completed", "cancelled", "failed"].includes(task.status);
  if (view === "attention") return task.status === "failed" || task.latest_execution?.status === "failed";
  return ["active", "paused"].includes(task.status);
}
const time = (value: string | null) => value ? new Date(value).toLocaleString() : "Not scheduled";

export function WorkOverview({ active, user, ownerId, initialTaskId = "", selectionRequest = 0, projects, users, onProject, onAgenda, onConversation, onQuestions }: {
  active: boolean; user: string; ownerId: string; initialTaskId?: string; selectionRequest?: number; projects: Project[]; users: Array<{ user_id: string; display_name: string }>; onProject: (project: Project) => void;
  onAgenda: (id: string) => void; onConversation: (day: string) => void; onQuestions: () => void;
}) {
  const [view, setView] = useState<WorkView>("active");
  const [page, setPage] = useState<WorkPage | null>(null);
  const [schedules, setSchedules] = useState<Schedule[]>([]);
  const [offset, setOffset] = useState(0);
  const [selected, setSelected] = useState<WorkItem | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");
  const [scheduleError, setScheduleError] = useState("");
  const request = useRef(0);
  const detail = useRef<HTMLElement>(null);
  const requestedTask = useRef(initialTaskId);
  useEffect(() => { requestedTask.current = initialTaskId; if (initialTaskId) { setView("active"); setOffset(0); } }, [initialTaskId, selectionRequest]);
  const memberName = (id: string) => users.find((member) => member.user_id === id)?.display_name || id;
  useEffect(() => { if (selected) detail.current?.focus({ preventScroll: false }); }, [selected?.task_id]);
  const load = useCallback(async () => {
    const id = ++request.current;
    if (!user || !ownerId) return;
    setLoading(true); setError(""); setScheduleError("");
    const [work, timed] = await Promise.allSettled([
      daemonRequest<WorkPage>("work_items", { actor_user_id: user, owner_user_id: ownerId, view, offset, limit: 50 }),
      daemonRequest<{ tasks: Schedule[] }>("scheduled_tasks", { actor_user_id: user, owner_user_id: ownerId, limit: 100 }),
    ]);
    if (id !== request.current) return;
    if (work.status === "fulfilled") {
      setPage(work.value);
      const requestedId = requestedTask.current;
      requestedTask.current = "";
      setSelected((current) => work.value.items.find((item) => item.task_id === (requestedId || current?.task_id)) || null);
    } else { setPage(null); setSelected(null); setError(String(work.reason?.message || "Work could not be loaded.")); }
    if (timed.status === "fulfilled") setSchedules(timed.value.tasks);
    else { setSchedules([]); setScheduleError(String(timed.reason?.message || "Schedules could not be loaded.")); }
    setLoading(false);
  }, [user, ownerId, view, offset, initialTaskId, selectionRequest]);
  useEffect(() => { if (active) void load(); return () => { request.current++; }; }, [load, active]);
  const chooseView = (next: WorkView) => { setView(next); setOffset(0); setSelected(null); };
  const items = page?.items || [];
  const visibleSchedules = schedules.filter((task) => scheduleMatches(task, view));
  const everyday = (projectId: string) => !projectId || projects.some((project) => project.project_id === projectId && project.is_system_home);
  const projectGroups = projects.filter((project) => !project.is_system_home && (
    view === "history" ? ["completed", "archived"].includes(project.status) || items.some((item) => item.project_id === project.project_id) || visibleSchedules.some((item) => item.project_id === project.project_id) :
    view === "attention" ? items.some((item) => item.project_id === project.project_id) || visibleSchedules.some((item) => item.project_id === project.project_id) : ["active", "paused"].includes(project.status) || items.some((item) => item.project_id === project.project_id) || visibleSchedules.some((item) => item.project_id === project.project_id)
  ));
  const taskRow = (item: WorkItem) => <button type="button" key={item.task_id} className={`work-task${selected?.task_id === item.task_id ? " selected" : ""}`} onClick={() => setSelected(item)}>
    <span><strong>{item.goal}</strong><small>Owner: {memberName(item.owner_user_id)} · Assigned: Alphonse</small><small>{item.next_action}</small></span>
    <span className="work-task-state"><span>{workStatus(item.status)}{item.blocked && !["done", "failed", "cancelled"].includes(item.status) ? " · Blocked" : ""}</span><small>{item.criteria.length ? `${item.completed_criteria} of ${item.criteria.length} criteria met` : "Progress not recorded"}</small></span>
  </button>;
  const scheduleRow = (task: Schedule) => <button type="button" className="work-task" key={task.scheduled_task_id} onClick={() => onAgenda(task.scheduled_task_id)}><span><strong>{task.name}</strong><small>Scheduled · Owner: {memberName(task.owner_user_id)}</small><small>{task.next_run_at ? `Next: ${time(task.next_run_at)}` : "No next run"}</small></span><span className="work-task-state">{workStatus(task.status)}<small>Latest execution: {task.latest_execution?.status || "Not run"}</small></span></button>;
  return <div className="work-overview">
    <header className="work-heading"><div><p className="eyebrow">The bigger plans. The everyday things.</p><h1>Projects &amp; Tasks</h1></div><button onClick={() => onAgenda("")}>Agenda</button></header>
    <div className="work-filters" role="group" aria-label="Work filter">{([["active", "Active work"], ["attention", "Needs input / blocked"], ["history", "Completed & history"]] as const).map(([value, label]) => <button key={value} aria-pressed={view === value} onClick={() => chooseView(value)}>{label}</button>)}<button disabled={loading} onClick={() => void load()}>Refresh</button></div>
    <p>Owner: {memberName(ownerId)}. Project totals cover recorded work; task lists are paged. Alphonse executes delegated tasks; questions may involve another member.</p>
    {loading ? <p role="status">Loading work…</p> : error ? <div role="alert"><p>{error}</p><button onClick={() => void load()}>Retry</button></div> : <>
      {scheduleError && <p role="alert">Schedules unavailable: {scheduleError} <button onClick={() => void load()}>Retry</button></p>}
      <h2>Projects</h2>
      {projectGroups.map((project) => {
        const nested = items.filter((item) => item.project_id === project.project_id);
        const scheduled = visibleSchedules.filter((item) => item.project_id === project.project_id);
        const totals = page?.project_summaries?.[project.project_id];
        const done = totals?.completed ?? nested.filter((item) => item.status === "done").length;
        const total = totals?.total ?? nested.length;
        const basis = totals ? "recorded" : "shown";
        return <details className="work-project" key={project.project_id} open={view === "attention" || undefined}><summary><span><strong>{project.name}</strong><small>{project.visibility === "shared" ? "Shared" : "Private"} · Owner: {memberName(project.owner_user_id)}</small><small>{project.description}</small><small>{nested.find((item) => item.status === "waiting_user" || item.blocked)?.next_action || nested[0]?.next_action || (scheduled.length ? "Review upcoming schedules in Agenda" : "Review project details or choose the next step in conversation")}</small></span><span>{workStatus(project.status)}<small>{total ? `${done} of ${total} ${basis} tasks completed` : "No tasks recorded"}</small>{total > 0 && <progress aria-label={`${project.name}: ${basis} task completion`} value={done} max={total} />}{totals && (totals.failed > 0 || totals.cancelled > 0) && <small>{totals.failed} failed · {totals.cancelled} cancelled</small>}</span></summary>
          <div className="work-project-tasks">{nested.map(taskRow)}{scheduled.map(scheduleRow)}{!nested.length && !scheduled.length && <p>No work in this view. Ask Alphonse in conversation to work on this project.</p>}<button onClick={() => onProject(project)}>Project details, sharing &amp; files</button></div>
        </details>;
      })}
      {!projectGroups.length && <p>No projects in this view.</p>}
      <h2>Everyday tasks</h2><p>No project needed.</p>
      <div className="work-everyday">{items.filter((item) => item.everyday).map(taskRow)}{visibleSchedules.filter((task) => everyday(task.project_id)).map(scheduleRow)}</div>
      {!items.some((item) => item.everyday) && !visibleSchedules.some((task) => everyday(task.project_id)) && <p>No everyday work in this view. Delegate a task in conversation.</p>}
      {items.some((item) => !item.everyday && !projects.some((project) => project.project_id === item.project_id)) && <section><h2>Other project work</h2>{items.filter((item) => !item.everyday && !projects.some((project) => project.project_id === item.project_id)).map(taskRow)}</section>}
      {visibleSchedules.some((task) => !everyday(task.project_id) && !projects.some((project) => project.project_id === task.project_id)) && <section><h2>Other scheduled work</h2>{visibleSchedules.filter((task) => !everyday(task.project_id) && !projects.some((project) => project.project_id === task.project_id)).map(scheduleRow)}</section>}
      <nav className="work-pagination" aria-label="Task pages"><button disabled={!offset} onClick={() => setOffset(Math.max(0, offset - 50))}>Previous tasks</button><span>Page {Math.floor(offset / 50) + 1}</span><button disabled={!page?.has_more} onClick={() => setOffset(offset + 50)}>Next tasks</button></nav>
      {schedules.length >= 100 && <p>Showing the latest 100 schedules. Open Agenda for schedule management.</p>}
      {selected && <section ref={detail} tabIndex={-1} className="work-detail" aria-label="Task details"><header className="work-heading"><h2>{selected.goal}</h2><button onClick={() => setSelected(null)}>Close details</button></header><p>{workStatus(selected.status)} · {selected.next_action}</p><p>Owner: {memberName(selected.owner_user_id)} · Execution: Alphonse{selected.waiting_for_user_id && <> · Waiting for: {memberName(selected.waiting_for_user_id)}</>}</p><h3>Acceptance progress</h3>{selected.criteria.length ? <ul>{selected.criteria.map((criterion) => <li key={criterion.id}>{criterion.statement} — {criterion.status}</li>)}</ul> : <p>No acceptance progress was recorded.</p>}<h3>Item history</h3><p>Created: {time(selected.created_at)}<br />Last persisted update: {time(selected.updated_at)}</p><p>Task ID: <code>{selected.task_id}</code>{selected.project_id && <> · Project ID: <code>{selected.project_id}</code></>}</p><div className="dialog-actions">{selected.conversation_day ? <><button onClick={() => onConversation(selected.conversation_day!)}>Related conversation ({selected.conversation_day})</button>{selected.status === "waiting_user" && selected.can_answer !== false && <button onClick={onQuestions}>Pending questions</button>}</> : <p>This owner’s conversation and answers stay in their own channel.</p>}</div><p>Task state is recorded by execution. To retry failed or cancelled work, review it with Alphonse in conversation.</p></section>}
    </>}
  </div>;
}
