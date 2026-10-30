import { Agenda } from "./Agenda";
import { useCallback, useEffect, useRef, useState, type RefObject } from "react";
import { PanelRightClose } from "lucide-react";
import { changeDrawer, reconcileContext, readDrawer, saveDrawer, widgetAvailability, WIDGET_CATALOG, type DrawerAction, type DrawerState } from "./atHandState";

import { authorizedWidgets, suggestedPlacements, type WidgetContext, type SkillWidget } from "./widgetContext";

export function useAtHand(member: string) {
  // An identity-tagged snapshot avoids writing the previous member's layout into a new key.
  const [snapshot, setSnapshot] = useState(() => ({ member, state: readDrawer(window.localStorage, member) }));
  const [saveFailed, setSaveFailed] = useState(false);
  const state = snapshot.member === member ? snapshot.state : readDrawer(window.localStorage, member);
  const act = (action: DrawerAction) => {
    if (!member) return;
    const next = changeDrawer(state, action);
    setSnapshot({ member, state: next });
    setSaveFailed(!saveDrawer(window.localStorage, member, next));
  };
  const memberRef = useRef(member);
  memberRef.current = member;
  const reconcile = useCallback((context: WidgetContext | undefined) => {
    const currentMember = memberRef.current;
    if (!context || context.member !== currentMember) return;
    const candidates = suggestedPlacements(authorizedWidgets(context, currentMember));
    setSnapshot((previous) => {
      const current = previous.member === currentMember ? previous.state : readDrawer(window.localStorage, currentMember);
      const next = reconcileContext(current, candidates, Date.now());
      if (next === current && previous.member === currentMember) return previous;
      setSaveFailed(!saveDrawer(window.localStorage, currentMember, next));
      return { member: currentMember, state: next };
    });
  }, []);
  return { state, act, saveFailed, reconcile };
}

export function AtHandDrawer({ state, act, member, connected, questionCount, questionsMember, onAttention, onAgenda, toggleRef, saveFailed, widgetContext, timezone, onSchedule, onProjects }: {
  timezone: string; onSchedule: (id: string) => void; onProjects: () => void;
  state: DrawerState; act: (action: DrawerAction) => void; member: string; connected: boolean;
  questionCount: number; questionsMember: string; onAttention: () => void; onAgenda: () => void;
  toggleRef: RefObject<HTMLButtonElement | null>; saveFailed: boolean; widgetContext?: WidgetContext;
}) {
  const rows = authorizedWidgets(widgetContext, member);
  const closeRef = useRef<HTMLButtonElement>(null);
  const [now, setNow] = useState(Date.now);
  useEffect(() => { if (state.open) closeRef.current?.focus(); }, [state.open]);
  useEffect(() => { const timer = window.setInterval(() => setNow(Date.now()), 1000); return () => window.clearInterval(timer); }, []);
  const close = () => { act({ type: "visibility", open: false }); toggleRef.current?.focus(); };
  return <aside id="at-hand" className="at-hand-drawer" aria-label="At hand" hidden={!state.open} onKeyDown={(event) => { if (event.key === "Escape") { event.stopPropagation(); close(); } }}>
    <header><h2>At hand</h2><button ref={closeRef} aria-label="Close At hand" onClick={close}><PanelRightClose /></button></header>
    {saveFailed && <p role="status">Layout could not be saved. Changes last for this session.</p>}
    <details className="widget-catalog"><summary>Add widget</summary><p>Choose a widget to keep at hand.</p>{WIDGET_CATALOG.map((entry) => <div key={entry.kind}><strong>{entry.title}</strong><p>{entry.description}</p><button disabled={!member || state.widgets.some((widget) => widget.id === entry.kind)} onClick={() => act({ type: "add", kind: entry.kind })}>Add {entry.title}</button></div>)}</details>
    <p className="widget-help">Pins stay until removed. Unpinned cards expire after 30 minutes. Unlock before moving or removing a locked card. Removing a card only changes this drawer.</p>
    <div className="at-hand-cards" aria-label="Widgets">
      {!state.widgets.length && <p>No widgets. Use Add widget to choose one.</p>}
      {state.widgets.map((widget, index) => {
        const data = rows.find((row) => row.id === widget.id);
        const title = data?.title || WIDGET_CATALOG.find((entry) => entry.kind === widget.kind)?.title || "Unavailable widget";
        const availability = widget.kind === "skill" && (!data || data.status !== "ready") ? "unavailable" : widgetAvailability(widget, member, connected, now);
        const locked = widget.mode === "locked";
        const agendaBody = <Agenda key={member} compact active={state.open} member={member} timezone={timezone} onSchedule={onSchedule} onProjects={onProjects} onOpen={onAgenda} />;
        return <article className="card at-hand-card" key={widget.id} aria-label={title}>
          <h3>{title}</h3><small className="widget-mode">{locked ? "User locked" : widget.mode === "pinned" ? "Pinned" : "Temporary context"}</small>
          <div className="widget-body" tabIndex={0} aria-label={`${title} content`}>
            {availability === "loading" ? <p role="status">Loading member…</p> : availability === "expired" ? <p role="status">Context expired. Remove this card or pin it to keep it.</p> : availability === "unavailable" ? <p role="status">Unavailable. Reconnect or check access in the workspace.</p> : widget.kind === "skill" ? data!.binding === "agenda" ? agendaBody : <SkillWidgetBody data={data!} onAgenda={onAgenda} /> : widget.kind === "questions" ? questionsMember !== member ? <p role="status">Loading questions…</p> : <><p>{questionCount ? `${questionCount} waiting for your input` : "No pending questions"}</p><button onClick={onAttention}>Needs your attention</button></> : agendaBody}
          </div>
          <div className="widget-controls" role="group" aria-label={`${title} controls`}>
            <button aria-label={`${locked ? "Unlock" : "Lock"} ${title}`} onClick={() => act({ type: "mode", id: widget.id, mode: locked ? "pinned" : "locked" })}>{locked ? "Unlock" : "Lock"}</button>
            {!locked && <button aria-label={`${widget.mode === "pinned" ? "Unpin" : "Pin"} ${title}`} onClick={() => act({ type: "mode", id: widget.id, mode: widget.mode === "pinned" ? "contextual" : "pinned", expiresAt: Date.now() + 30 * 60 * 1000 })}>{widget.mode === "pinned" ? "Unpin" : "Pin"}</button>}
            <button aria-label={`Move ${title} up`} disabled={locked || index === 0 || state.widgets[index - 1]?.mode === "locked"} onClick={() => act({ type: "move", id: widget.id, direction: -1 })}>↑</button>
            <button aria-label={`Move ${title} down`} disabled={locked || index === state.widgets.length - 1 || state.widgets[index + 1]?.mode === "locked"} onClick={() => act({ type: "move", id: widget.id, direction: 1 })}>↓</button>
            <button aria-label={`Remove ${title}`} disabled={locked} onClick={() => act({ type: "remove", id: widget.id })}>Remove</button>
          </div>
        </article>;
      })}
    </div>
  </aside>;
}

function SkillWidgetBody({ data, onAgenda }: { data: SkillWidget; onAgenda: () => void }) {
  return <>{data.historical && <small>Historical context · not a current measurement</small>}<p style={{ whiteSpace: "pre-wrap" }}>{data.text}</p>
    {data.binding === "agenda" && <button onClick={onAgenda}>Open Agenda</button>}
    {data.relevant && ["pin", "remove", "expire"].includes(data.suggestion) && <p role="status">Alphonse suggests {data.suggestion === "pin" ? "keeping this card pinned" : data.suggestion === "remove" ? "removing this card" : "letting this context expire"}. Use the card controls to apply.</p>}
  </>;
}
