import { useCallback, useEffect, useRef, useState } from "react";
import { ArrowRight, CalendarDays, ChevronDown, Folder, Plus, RefreshCw } from "lucide-react";
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
export type ProjectTodo = { todo_id: string; title: string; status: "pending" | "completed" | "cancelled" };
type TodoPage = { todos: ProjectTodo[]; todos_more: boolean };
export async function loadProjectTodos(user: string, ownerId: string, projectId: string): Promise<ProjectTodo[]> {
  const todos: ProjectTodo[] = [];
  let more = true;
  while (more) {
    const page = await daemonRequest<TodoPage>("project_records", { actor_user_id: user, owner_user_id: ownerId, project_id: projectId, todos_offset: todos.length, limit: 100 });
    todos.push(...page.todos);
    more = page.todos_more && page.todos.length > 0;
  }
  return todos;
}
export function todoMatches(todo: ProjectTodo, view: WorkView) {
  return view === "active" ? todo.status === "pending" : view === "history" ? todo.status !== "pending" : false;
}
export function projectProgress(todos: ProjectTodo[], total: number, completed: number) {
  return { total: total + todos.filter(todo => todo.status !== "cancelled").length, completed: completed + todos.filter(todo => todo.status === "completed").length };
}
export function ProjectTodoRow({ todo, onOpen }: { todo: ProjectTodo; onOpen: () => void }) {
  return <button type="button" className="work-task" onClick={onOpen}><span className={`work-task-indicator${todo.status === "completed" ? " done" : ""}`} aria-hidden="true">{todo.status === "completed" ? "✓" : ""}</span><span className={`work-task-title${todo.status === "completed" ? " done" : ""}`}>{todo.title}</span><span className="work-task-state">To-do · {workStatus(todo.status)}</span></button>;
}
type Schedule = { scheduled_task_id: string; project_id: string; owner_user_id: string; name: string; status: string; next_run_at: string | null; latest_execution?: { status: string } | null };
export type WorkView = "active" | "attention" | "history";
export const workStatus = (status: string) => ({ done: "Completed", pending: "Pending", waiting_user: "Waiting for input", queued: "Queued", running: "Running", failed: "Failed", cancelled: "Cancelled", active: "Active", paused: "Paused", completed: "Completed", archived: "Archived" }[status] || status);
export function scheduleMatches(task: Schedule, view: WorkView) {
  if (view === "history") return ["completed", "cancelled", "failed"].includes(task.status);
  if (view === "attention") return task.status === "failed" || task.latest_execution?.status === "failed";
  return ["active", "paused"].includes(task.status);
}
const time = (value: string | null) => value ? new Date(value).toLocaleString() : "Not scheduled";

export function WorkOverview({ onCreate, onOwnerChange, active, user, ownerId, initialTaskId = "", selectionRequest = 0, refreshKey = 0, projects, users, onProject, onAgenda, onConversation, onQuestions }: {
  onCreate: () => void; onOwnerChange: (id: string) => void; active: boolean; user: string; ownerId: string; initialTaskId?: string; selectionRequest?: number; refreshKey?: number; projects: Project[]; users: Array<{ user_id: string; display_name: string }>; onProject: (project: Project) => void;
  onAgenda: (id: string) => void; onConversation: (day: string) => void; onQuestions: () => void;
}) {
  const [view, setView] = useState<WorkView>("active");
  const [page, setPage] = useState<WorkPage | null>(null);
  const [todos, setTodos] = useState<Record<string, ProjectTodo[]>>({});
  const [todoError, setTodoError] = useState("");
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
  const projectIds = projects.filter(project => !project.is_system_home).map(project => project.project_id).sort().join("\0");
  const load = useCallback(async () => {
    const id = ++request.current;
    if (!user || !ownerId) return;
    setLoading(true); setError(""); setScheduleError(""); setTodoError("");
    const [work, timed, projectTodos] = await Promise.allSettled([
      daemonRequest<WorkPage>("work_items", { actor_user_id: user, owner_user_id: ownerId, view, offset, limit: 50 }),
      daemonRequest<{ tasks: Schedule[] }>("scheduled_tasks", { actor_user_id: user, owner_user_id: ownerId, limit: 100 }),
      Promise.all((projectIds ? projectIds.split("\0") : []).map(async projectId => [projectId, await loadProjectTodos(user, ownerId, projectId)] as const)),
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
    if (projectTodos.status === "fulfilled") setTodos(Object.fromEntries(projectTodos.value));
    else { setTodos({}); setTodoError(String(projectTodos.reason?.message || "Project to-dos could not be loaded.")); }
    setLoading(false);
  }, [user, ownerId, view, offset, initialTaskId, selectionRequest, projectIds, refreshKey]);
  useEffect(() => { if (active) void load(); return () => { request.current++; }; }, [load, active]);
  const chooseView = (next: WorkView) => { setView(next); setOffset(0); setSelected(null); };
  const items = page?.items || [];
  const visibleSchedules = schedules.filter((task) => scheduleMatches(task, view));
  const everyday = (projectId: string) => !projectId || projects.some((project) => project.project_id === projectId && project.is_system_home);
  const projectGroups = projects.filter((project) => !project.is_system_home && (
    view === "history" ? ["completed", "archived"].includes(project.status) || items.some((item) => item.project_id === project.project_id) || visibleSchedules.some((item) => item.project_id === project.project_id) || (todos[project.project_id] || []).some(todo => todoMatches(todo, view)) :
    view === "attention" ? items.some((item) => item.project_id === project.project_id) || visibleSchedules.some((item) => item.project_id === project.project_id) || (todos[project.project_id] || []).some(todo => todoMatches(todo, view)) : ["active", "paused"].includes(project.status) || items.some((item) => item.project_id === project.project_id) || visibleSchedules.some((item) => item.project_id === project.project_id) || (todos[project.project_id] || []).some(todo => todoMatches(todo, view))
  ));
  const taskRow = (item: WorkItem) => <button type="button" key={item.task_id} className={`work-task${selected?.task_id === item.task_id ? " selected" : ""}`} onClick={() => setSelected(item)}>
    <span className={`work-task-indicator${item.status === "done" ? " done" : ""}`} aria-hidden="true">{item.status === "done" ? "✓" : ""}</span><span className={`work-task-title${item.status === "done" ? " done" : ""}`}>{item.goal}</span>
    <span className={`work-task-state${item.status === "waiting_user" || item.blocked ? " attention" : ""}`}>{workStatus(item.status)}{item.blocked && !["done", "failed", "cancelled"].includes(item.status) ? " · Blocked" : ""}</span>
  </button>;
  const scheduleRow = (task: Schedule) => <button type="button" className="work-task" key={task.scheduled_task_id} onClick={() => onAgenda(task.scheduled_task_id)}><span className="work-task-indicator" aria-hidden="true"><CalendarDays size={12} /></span><span className="work-task-title">{task.name}</span><span className="work-task-state">Scheduled · {workStatus(task.status)}</span></button>;
  return <div className="work-overview">
    <header className="work-heading work-overview-heading"><div><h1>Projects &amp; Tasks</h1><p>The bigger plans. The everyday things.</p></div><button className="work-add" onClick={onCreate}><Plus size={15} /> Add</button></header>
    <div className="work-controls"><div className="work-filters" role="group" aria-label="Work filter">{([["active", "In progress"], ["attention", "Needs you"], ["history", "History"]] as const).map(([value, label]) => <button key={value} aria-pressed={view === value} onClick={() => chooseView(value)}>{label}</button>)}</div><button className="work-agenda" onClick={() => onAgenda("")}><CalendarDays size={15} /> Agenda</button></div>
    <div className="work-utility"><label>Owner <select aria-label="Work owner" value={ownerId} onChange={event => onOwnerChange(event.target.value)}>{users.map(member => <option key={member.user_id} value={member.user_id}>{member.display_name}</option>)}</select></label><button disabled={loading} onClick={() => void load()} aria-label="Refresh work" title="Refresh work"><RefreshCw size={14} /></button></div>
    {loading ? <p role="status">Loading work…</p> : error ? <div role="alert"><p>{error}</p><button onClick={() => void load()}>Retry</button></div> : <>
      {todoError && <p role="alert">Project to-dos unavailable: {todoError} <button onClick={() => void load()}>Retry</button></p>}
      {scheduleError && <p role="alert">Schedules unavailable: {scheduleError} <button onClick={() => void load()}>Retry</button></p>}
      <div className="work-section-heading"><h2>Projects</h2><span>{projectGroups.length} {projectGroups.length === 1 ? "project" : "projects"}</span></div>
      {projectGroups.map((project, index) => {
        const nested = items.filter((item) => item.project_id === project.project_id);
        const scheduled = visibleSchedules.filter((item) => item.project_id === project.project_id);
        const projectTodos = todos[project.project_id] || [];
        const visibleTodos = projectTodos.filter(todo => todoMatches(todo, view));
        const totals = page?.project_summaries?.[project.project_id];
        const { completed: done, total } = projectProgress(projectTodos, totals?.total ?? nested.length, totals?.completed ?? nested.filter((item) => item.status === "done").length);
        const basis = totals ? "recorded" : "shown";
        return <details className="work-project" key={project.project_id} open={view === "attention" || index === 0 || undefined}><summary><span className="work-project-icon"><Folder size={17} /></span><span className="work-project-name"><strong>{project.name}</strong><small>{project.visibility === "shared" ? "Shared" : "Private"} · {memberName(project.owner_user_id)}</small></span><span className="work-project-progress"><small>{todoError ? "Task count unavailable" : total ? `${done} of ${total} complete` : "No tasks recorded"}</small>{!todoError && total > 0 && <progress aria-label={`${project.name}: ${basis} task completion`} value={done} max={total} />}<small>{workStatus(project.status)}</small></span><ChevronDown className="work-project-chevron" size={15} /></summary>
          <div className="work-project-tasks">{nested.map(taskRow)}{visibleTodos.map(todo => <ProjectTodoRow key={todo.todo_id} todo={todo} onOpen={() => onProject(project)} />)}{scheduled.map(scheduleRow)}{!todoError && !nested.length && !scheduled.length && !visibleTodos.length && <p>No work in this view. Ask Alphonse in conversation to work on this project.</p>}{page?.has_more && <p>More tasks may be available on other pages.</p>}<button className="work-project-link" onClick={() => onProject(project)}>View project details <ArrowRight size={14} /></button></div>
        </details>;
      })}
      {!projectGroups.length && <p>No projects in this view.</p>}
      <div className="work-section-heading"><h2>Everyday tasks</h2><span>No project needed</span></div>
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
