import { useEffect, useRef, useState } from "react";
import { X } from "lucide-react";
import { daemonRequest } from "./api";
import { workStatus, type WorkItem } from "./WorkOverview";
import type { Project } from "./types";

type Page = { items: WorkItem[]; has_more: boolean; offset: number };
type ActivityView = "active" | "history";

export function ProjectActivityDialog({ project, user, ownerId, ownerName, onClose, onSettings, onConversation, onQuestions }: {
  project: Project; user: string; ownerId: string; ownerName: string; onClose: () => void; onSettings: () => void;
  onConversation: (day: string) => void; onQuestions: () => void;
}) {
  const dialog = useRef<HTMLDialogElement>(null);
  const [pages, setPages] = useState<Record<ActivityView, Page | null>>({ active: null, history: null });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [revision, setRevision] = useState(0);
  const [offsets, setOffsets] = useState<Record<ActivityView, number>>({ active: 0, history: 0 });
  useEffect(() => { dialog.current?.showModal(); return () => { dialog.current?.close(); }; }, []);
  useEffect(() => {
    let live = true;
    setLoading(true); setError("");
    Promise.all((["active", "history"] as const).map(view => daemonRequest<Page>("work_items", {
      actor_user_id: user, owner_user_id: ownerId, project_id: project.project_id, view, offset: offsets[view], limit: 50,
    }))).then(([active, history]) => { if (live) setPages({ active, history }); })
      .catch(cause => { if (live) { setPages({ active: null, history: null }); setError(cause instanceof Error ? cause.message : "Project work could not be loaded."); } })
      .finally(() => { if (live) setLoading(false); });
    return () => { live = false; };
  }, [user, ownerId, project.project_id, offsets, revision]);
  const items = [...(pages.active?.items || []), ...(pages.history?.items || [])].sort((a, b) => b.updated_at.localeCompare(a.updated_at));
  return <dialog ref={dialog} className="project-activity-dialog" aria-labelledby="project-activity-title" onCancel={event => { event.preventDefault(); onClose(); }}>
    <header><div><h1 id="project-activity-title">{project.name}</h1><p>{project.visibility === "shared" ? "Shared" : "Personal"} · {ownerName}</p></div><button className="project-activity-close" onClick={onClose} aria-label="Close project details"><X size={19} /></button></header>
    <div className="project-activity-heading"><h2>Activity &amp; history</h2><span>{workStatus(project.status)}</span></div>
    {loading ? <p role="status">Loading project work…</p> : error ? <div role="alert"><p>{error}</p><button onClick={() => setRevision(value => value + 1)}>Retry</button></div> : <>
      {items.length ? <ol className="project-activity-list">{items.map(item => <li key={item.task_id}>
        <strong>{item.goal}</strong><span>{workStatus(item.status)}{item.blocked ? " · Blocked" : ""} · Updated {new Date(item.updated_at).toLocaleString()}</span><p>{item.next_action}</p>
        <div className="project-activity-actions">{item.conversation_day && <button onClick={() => onConversation(item.conversation_day!)}>Related conversation</button>}{item.status === "waiting_user" && item.can_answer && <button onClick={onQuestions}>Answer pending question</button>}</div>
      </li>)}</ol> : <p>No recorded work for {ownerName} on these pages.</p>}
      {(["active", "history"] as const).map(view => <nav className="project-activity-pages" key={view} aria-label={`${view === "active" ? "Active work" : "Completed work"} pages`}><span>{view === "active" ? "Active work" : "Completed and past work"}</span><button disabled={!offsets[view]} onClick={() => setOffsets(current => ({ ...current, [view]: Math.max(0, current[view] - 50) }))}>Previous</button><button disabled={!pages[view]?.has_more} onClick={() => setOffsets(current => ({ ...current, [view]: current[view] + 50 }))}>Next</button></nav>)}
      <p className="project-activity-note">Showing the latest recorded state of {ownerName}’s delegated work. Open a related conversation for more detail.</p>
    </>}
    <div className="project-activity-actions"><button onClick={() => onConversation("")}>Continue in conversation</button><button className="project-activity-settings" onClick={onSettings}>Project settings &amp; files</button></div>
  </dialog>;
}
