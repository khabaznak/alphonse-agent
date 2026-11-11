import { useEffect, useRef, useState } from "react";
import { CalendarClock, X } from "lucide-react";
import { daemonRequest } from "./api";
import { workStatus } from "./WorkOverview";
import type { Project } from "./types";

export type Records = {
  todos: Array<{ todo_id: string; title: string; status: "pending" | "completed" | "cancelled" }>; todos_more: boolean;
  schedules: Array<{ scheduled_task_id: string; name: string; status: string; next_run_at: string | null; schedule_kind: string }>;
  schedules_more: boolean; executions: Array<{ run_id: string; scheduled_task_id: string; task_name: string; status: string; started_at: string }>;
  executions_more: boolean;
};

export function ProjectActivityDialog({ project, user, ownerId, ownerName, onClose, onSettings, onDelegate, onAgenda, onUpdated, onDeleted }: {
  project: Project; user: string; ownerId: string; ownerName: string; onClose: () => void; onSettings: () => void;
  onDelegate: () => void; onAgenda: (id: string) => void; onUpdated: () => void; onDeleted: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [records, setRecords] = useState<Records | null>(null);
  const [recordError, setRecordError] = useState("");
  const [recordLoading, setRecordLoading] = useState(true);
  const [recordOffsets, setRecordOffsets] = useState({ schedules_offset: 0, executions_offset: 0, todos_offset: 0 });
  const [todoTitle, setTodoTitle] = useState("");
  const [todoKey, setTodoKey] = useState(() => crypto.randomUUID());
  const [todoBusy, setTodoBusy] = useState(false);
  const [todoError, setTodoError] = useState("");
  const [editing, setEditing] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [name, setName] = useState(project.name);
  const [description, setDescription] = useState(project.description);
  const [confirmation, setConfirmation] = useState("");
  const [saving, setSaving] = useState(false);
  const [actionError, setActionError] = useState("");
  const [revision, setRevision] = useState(0);
  useEffect(() => { dialog.current?.showModal(); return () => { dialog.current?.close(); }; }, []);
  useEffect(() => {
    let live = true; setRecordLoading(true); setRecordError("");
    daemonRequest<Records>("project_records", { actor_user_id: user, owner_user_id: ownerId, project_id: project.project_id, ...recordOffsets, limit: 50 })
      .then(result => { if (live) setRecords(result); })
      .catch(cause => { if (live) { setRecords(null); setRecordError(cause instanceof Error ? cause.message : "Project records could not be loaded."); } })
      .finally(() => { if (live) setRecordLoading(false); });
    return () => { live = false; };
  }, [user, ownerId, project.project_id, recordOffsets, revision]);
  const mutateTodo = async (action: "create" | "complete" | "reopen" | "cancel", todoId = "") => {
    setTodoBusy(true); setTodoError("");
    try {
      await daemonRequest("mutate_project_todo", { project_id: project.project_id, action, todo_id: todoId, title: todoTitle.trim(), request_key: todoKey });
      if (action === "create") { setTodoTitle(""); setTodoKey(crypto.randomUUID()); setRecordOffsets(current => ({ ...current, todos_offset: 0 })); }
      setRevision(value => value + 1);
    } catch (cause) { setTodoError(cause instanceof Error ? cause.message : "To-do could not be saved."); }
    finally { setTodoBusy(false); }
  };
  const update = async () => {
    setSaving(true); setActionError("");
    try { await daemonRequest("update_project", { user, project_id: project.project_id, name: name.trim(), description, visibility: project.visibility }); setEditing(false); onUpdated(); }
    catch (cause) { setActionError(cause instanceof Error ? cause.message : "Project could not be updated."); }
    finally { setSaving(false); }
  };
  const remove = async () => {
    setSaving(true); setActionError("");
    try { await daemonRequest("delete_project", { user, project_id: project.project_id, confirmation }); onDeleted(); }
    catch (cause) { const message = cause instanceof Error ? cause.message : "Project could not be deleted."; setActionError(message.includes("project_has_active_scheduled_tasks") ? "Cancel the project’s active or paused schedules before deleting it." : message); }
    finally { setSaving(false); }
  };
  return <dialog ref={dialog} className="project-activity-dialog" aria-labelledby="project-activity-title" onCancel={event => { event.preventDefault(); onClose(); }}>
    <header><h1 id="project-activity-title">{project.name}</h1><button className="project-activity-close" onClick={onClose} aria-label="Close project details"><X size={19} /></button></header>
    <div className="project-activity-body">
    <p className="project-activity-meta">{project.visibility === "shared" ? "Shared" : "Personal"} · {ownerName} · {workStatus(project.status)}</p>
    {project.description && <p className="project-activity-description">{project.description}</p>}
    <h2 className="project-records-title">To-dos</h2>
    {recordLoading ? <p role="status">Loading project to-dos…</p> : recordError ? <p>Project to-dos are unavailable.</p> : records && <>
      <ProjectTodoList items={records.todos} editable={user === ownerId && project.status === "active"} busy={todoBusy} onChange={(id, action) => void mutateTodo(action, id)} />
      <nav className="project-activity-pages" aria-label="To-do pages"><span>To-dos</span><button disabled={!recordOffsets.todos_offset} onClick={() => setRecordOffsets(current => ({ ...current, todos_offset: Math.max(0, current.todos_offset - 50) }))}>Previous</button><button disabled={!records.todos_more} onClick={() => setRecordOffsets(current => ({ ...current, todos_offset: current.todos_offset + 50 }))}>Next</button></nav>
    </>}
    {user === ownerId && project.status === "active" && <form className="project-todo-add" onSubmit={event => { event.preventDefault(); if (todoTitle.trim()) void mutateTodo("create"); }}><label htmlFor="project-todo-title">New to-do</label><input id="project-todo-title" value={todoTitle} onChange={event => setTodoTitle(event.target.value)} maxLength={300} placeholder="Add a to-do" /><button disabled={todoBusy || !todoTitle.trim()}>Add</button></form>}
    {todoError && <p role="alert">{todoError}</p>}
    <h2 className="project-records-title">Scheduled work</h2>
    {recordLoading ? <p role="status">Loading scheduled work…</p> : recordError ? <div role="alert"><p>{recordError}</p><button onClick={() => setRevision(value => value + 1)}>Retry</button></div> : records && <>
      <ProjectScheduleList items={records.schedules} onAgenda={onAgenda} />
      <nav className="project-activity-pages" aria-label="Scheduled work pages"><span>Scheduled work</span><button disabled={!recordOffsets.schedules_offset} onClick={() => setRecordOffsets(current => ({ ...current, schedules_offset: Math.max(0, current.schedules_offset - 50) }))}>Previous</button><button disabled={!records.schedules_more} onClick={() => setRecordOffsets(current => ({ ...current, schedules_offset: current.schedules_offset + 50 }))}>Next</button></nav>
    </>}
    <details className="project-history"><summary>Schedule execution history</summary>
      {recordLoading ? <p role="status">Loading schedule history…</p> : recordError ? <p>Schedule history is unavailable.</p> : records && <>{records.executions.length ? <ul className="project-record-list">{records.executions.map(run => <li key={run.run_id}><strong>{run.task_name}</strong><small>{workStatus(run.status)} · {new Date(run.started_at).toLocaleString()}</small></li>)}</ul> : <p>No schedule executions on this page.</p>}
      <nav className="project-activity-pages" aria-label="Schedule execution pages"><span>Schedule executions</span><button disabled={!recordOffsets.executions_offset} onClick={() => setRecordOffsets(current => ({ ...current, executions_offset: Math.max(0, current.executions_offset - 50) }))}>Previous</button><button disabled={!records.executions_more} onClick={() => setRecordOffsets(current => ({ ...current, executions_offset: current.executions_offset + 50 }))}>Next</button></nav></>}
    </details>
    <div className="project-activity-actions project-main-actions"><button onClick={onDelegate} disabled={project.status !== "active"}>Delegate work</button><button onClick={() => { setEditing(true); setDeleting(false); setActionError(""); }} disabled={project.status === "archived"}>Edit details</button><button onClick={() => { setDeleting(true); setEditing(false); setActionError(""); }}>Delete project</button><button onClick={onSettings}>Settings &amp; files</button></div>
    {editing && <form className="project-inline-edit" onSubmit={event => { event.preventDefault(); void update(); }}><label>Name<input value={name} onChange={event => setName(event.target.value)} required /></label><label>Description<textarea value={description} onChange={event => setDescription(event.target.value)} rows={2} /></label><div className="project-activity-actions"><button type="submit" disabled={saving || !name.trim()}>Save changes</button><button type="button" onClick={() => setEditing(false)}>Cancel</button></div></form>}
    {deleting && <div className="project-inline-edit"><p>Deleting removes this project from Alphonse and deletes its managed project folder and files. Imported folders are kept. Active or paused schedules must be cancelled first.</p><label>Type the project ID to confirm<input value={confirmation} onChange={event => setConfirmation(event.target.value)} /></label><code>{project.project_id}</code><div className="project-activity-actions"><button disabled={saving || confirmation !== project.project_id} onClick={() => void remove()}>Delete project permanently</button><button onClick={() => setDeleting(false)}>Cancel</button></div></div>}
    {actionError && <p role="alert">{actionError}</p>}
    </div>
  </dialog>;
}

export function ProjectScheduleList({ items, onAgenda }: { items: Records["schedules"]; onAgenda: (id: string) => void }) {
  if (!items.length) return <p className="project-work-empty">No scheduled work on this page.</p>;
  return <ul className="project-work-list">{items.map(task => <li key={task.scheduled_task_id}>
    <span className="project-work-icon"><CalendarClock size={16} /></span><span className="project-work-content"><button onClick={() => onAgenda(task.scheduled_task_id)}>{task.name}</button><small>{task.schedule_kind === "rrule" ? "Recurring" : task.schedule_kind === "once" ? "One-time" : "Scheduled"}{task.next_run_at ? ` · Next ${new Date(task.next_run_at).toLocaleString()}` : " · No next run"}</small></span><span className="project-work-status">{workStatus(task.status)}</span>
  </li>)}</ul>;
}

export function ProjectTodoList({ items, editable, busy, onChange }: { items: Records["todos"]; editable: boolean; busy: boolean; onChange: (id: string, action: "complete" | "reopen") => void }) {
  if (!items.length) return <p className="project-work-empty">No project to-dos on this page.</p>;
  return <ul className="project-work-list">{items.map(todo => <li key={todo.todo_id}><label className="project-todo-row"><input type="checkbox" checked={todo.status === "completed"} disabled={busy || !editable || todo.status === "cancelled"} onChange={() => onChange(todo.todo_id, todo.status === "completed" ? "reopen" : "complete")} /><span className={todo.status === "completed" ? "project-todo-complete" : ""}>{todo.title}</span></label><span className="project-work-status">{workStatus(todo.status)}</span></li>)}</ul>;
}
