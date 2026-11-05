import { useCallback, useEffect, useRef, useState } from "react";
import { daemonRequest } from "./api";
import { workStatus } from "./WorkOverview";

type Plan = { scheduled_task_id: string; name: string; status: string; next_run_at: string | null };
export type HouseholdOverview = {
  member: { user_id: string; display_name: string }; day: string; timezone: string; generated_at: string;
  projects: Array<{ project_id: string; name: string; status: string }>;
  plans: Plan[]; routines: Plan[];
  tasks: Array<{ task_id: string; goal: string; status: string; conversation_day: string }>;
  questions: Array<{ question_id: string; message: string }>;
  unavailable: string[]; bounded: string[];
};

// Validate both request identity and response audience before rendering any records.
export function currentHousehold(data: HouseholdOverview | null, user: string, personal: boolean, key: string): HouseholdOverview | null {
  return data?.member.user_id === user && key === `${user}:${personal}` ? data : null;
}

type Props = {
  user: string; active: boolean; connected: boolean; isAdmin: boolean;
  onProject: (id: string) => void; onAgenda: (id: string) => void;
  onConversation: (day: string) => void; onQuestions: () => void; onMembers: () => void;
};

export function HouseholdWorkspace({ user, active, connected, isAdmin, onProject, onAgenda, onConversation, onQuestions, onMembers }: Props) {
  const [personal, setPersonal] = useState(false);
  const [snapshot, setSnapshot] = useState<{ data: HouseholdOverview; key: string } | null>(null);
  const [error, setError] = useState(false);
  const [loading, setLoading] = useState(true);
  const generation = useRef(0);
  const requestKey = `${user}:${personal}`;
  const load = useCallback(async () => {
    const version = ++generation.current;
    setSnapshot(null); setError(false); setLoading(true);
    if (!active || !connected || !user || document.hidden || !document.hasFocus()) return;
    try {
      const data = await daemonRequest<HouseholdOverview>("household_overview", { actor_user_id: user, include_personal: personal });
      if (data.member.user_id !== user) throw new Error("Household audience mismatch");
      if (version === generation.current) setSnapshot({ data, key: `${user}:${personal}` });
    } catch { if (version === generation.current) setError(true); }
    finally { if (version === generation.current) setLoading(false); }
  }, [user, personal, active, connected]);
  useEffect(() => {
    void load();
    if (!active || !connected || !user) return () => { ++generation.current; };
    const clear = () => { ++generation.current; setSnapshot(null); };
    const visibility = () => { if (document.hidden) clear(); else void load(); };
    window.addEventListener("blur", clear); window.addEventListener("focus", load);
    document.addEventListener("visibilitychange", visibility);
    const timer = window.setInterval(() => void load(), 30_000);
    return () => { clear(); window.clearInterval(timer); window.removeEventListener("blur", clear); window.removeEventListener("focus", load); document.removeEventListener("visibilitychange", visibility); };
  }, [load, active, connected, user]);
  // Leaving Household resets private disclosure, including when navigating to a card.
  useEffect(() => { if (!active) setPersonal(false); }, [active]);
  const data = active && connected ? currentHousehold(snapshot?.data || null, user, personal, snapshot?.key || "") : null;
  const open = async (kind: "projects" | "plans" | "routines" | "tasks" | "questions", id: string) => {
    const version = ++generation.current;
    setSnapshot(null); setLoading(true);
    try {
      // Recheck membership, ownership and pending status at the moment a card is opened.
      const fresh = await daemonRequest<HouseholdOverview>("household_overview", { actor_user_id: user, include_personal: personal });
      if (version !== generation.current || fresh.member.user_id !== user) return;
      setSnapshot({ data: fresh, key: requestKey });
      if (kind === "projects" && fresh.projects.some(p => p.project_id === id)) onProject(id);
      else if ((kind === "plans" || kind === "routines") && fresh[kind].some(p => p.scheduled_task_id === id)) onAgenda(id);
      else if (kind === "tasks") { const task = fresh.tasks.find(t => t.task_id === id); if (task) onConversation(task.conversation_day); }
      else if (kind === "questions" && fresh.questions.some(q => q.question_id === id)) onQuestions();
    } catch { if (version === generation.current) setError(true); }
    finally { if (version === generation.current) setLoading(false); }
  };
  const plan = (item: Plan, kind: "plans" | "routines") => <li key={item.scheduled_task_id}><div><strong>{item.name}</strong><small>Only you · {item.next_run_at ? new Date(item.next_run_at).toLocaleString(undefined, { timeZone: data?.timezone }) : "No next run recorded"}</small></div><button onClick={() => void open(kind, item.scheduled_task_id)}>Open in Agenda</button></li>;
  return <div className="household-workspace">
    <header className="household-intro"><p className="eyebrow">A shared everyday life</p><h1>Home, in the picture.</h1><p>Plans, routines, and things that need a hand.</p><p>Current member: <strong>{data?.member.display_name || user}</strong> · {personal ? "Only you: your records alongside shared efforts" : "Shared: efforts you own or belong to"}</p>
      <div className="household-actions" role="group" aria-label="Household audience"><button aria-pressed={!personal} onClick={() => setPersonal(false)}>Shared</button><button aria-pressed={personal} onClick={() => setPersonal(true)}>Only you</button><button disabled={loading || !connected} onClick={() => void load()}>Refresh</button>{isAdmin && <button onClick={onMembers}>Manage members</button>}</div>
    </header>
    {!connected ? <p role="status">Household is unavailable while disconnected. Reconnect to refresh.</p> : error ? <div role="alert"><p>Household could not be refreshed. Previous cards have been cleared.</p><button onClick={() => void load()}>Retry</button></div> : !data ? <p role="status">Refreshing household…</p> : <>
      {!!data.unavailable.length && <p role="alert">Some sources are unavailable: {data.unavailable.join(", ")}. Their cards have been cleared. <button onClick={() => void load()}>Retry sources</button></p>}
      <div className="household-grid">
        <section className="card household-wide"><header><h2>Today at home</h2><span>{data.day} · {data.timezone}</span></header>{personal ? <>{data.plans.length ? <ul>{data.plans.map(item => plan(item, "plans"))}</ul> : <p>{data.unavailable.includes("schedules") ? "Today's plans are unavailable." : "No plans due today in your available schedules."}</p>}<button onClick={() => onAgenda("")}>View your Agenda</button></> : <p>Schedules are personal under the current sharing rules. Choose “Only you” to see your plans and routines.</p>}</section>
        <section className="card household-wide"><header><h2>Shared efforts</h2><span>Shared with project members</span></header>{data.projects.length ? <ul className="household-efforts">{data.projects.map(p => <li key={p.project_id}><div><strong>{p.name}</strong><small>Shared · {workStatus(p.status)}</small></div><button onClick={() => void open("projects", p.project_id)}>See the plan</button></li>)}</ul> : <p>{data.unavailable.includes("projects") ? "Shared efforts are unavailable." : "No active shared efforts for this member. Plan something together in conversation."}</p>}</section>
        {personal && <>
          <section className="card"><header><h2>Needs your input</h2><span>Only you</span></header>{data.questions.length ? <ul>{data.questions.map(q => <li key={q.question_id}><div><strong>{q.message}</strong><small>Your answer is requested</small></div><button onClick={() => void open("questions", q.question_id)}>Open question</button></li>)}</ul> : <p>{data.unavailable.includes("questions") ? "Questions are unavailable." : "No pending questions for you."}</p>}</section>
          <section className="card"><header><h2>Your routines</h2><span>Only you</span></header>{data.routines.length ? <ul>{data.routines.map(item => plan(item, "routines"))}</ul> : <p>{data.unavailable.includes("schedules") ? "Routines are unavailable." : "No active repeating schedules in this view."}</p>}</section>
          <section className="card household-wide"><header><h2>Small things, in progress</h2><span>Only you · Alphonse executes</span></header>{data.tasks.length ? <ul>{data.tasks.map(t => <li key={t.task_id}><div><strong>{t.goal}</strong><small>{workStatus(t.status)}</small></div><button onClick={() => void open("tasks", t.task_id)}>Related conversation</button></li>)}</ul> : <p>{data.unavailable.includes("tasks") ? "Tasks are unavailable." : "No active tasks in this view. Delegate everyday work in conversation."}</p>}</section>
        </>}
      </div>
      <p className="household-footnote">{personal ? "Up to 12 cards per section, from the first 100 active schedules and 50 active tasks. Open Agenda or Projects & Tasks for more. " : "Up to 12 active shared projects. "}Refreshed {new Date(data.generated_at).toLocaleTimeString()}. {data.bounded.length > 0 && "More available records may be found in the related workspace."}</p>
    </>}
  </div>;
}
